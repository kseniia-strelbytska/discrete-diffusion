"""
eval_oracle_param_sweep.py
==========================
Evaluate the oracle across a grid of:
  - grammar  : ["baN", "bbaN", "aNbN", "aNbNcN"]
  - L        : sequence length
  - strategy : ["uniform", "gaussian", "ar", "ebsampler"]
  - sampler  : ["greedy", "categorical"]
  - per-strategy hyperparameter (sweep only where meaningful):
        uniform   → none
        gaussian  → sigma ∈ GAUSSIAN_SIGMAS
        ar        → none
        ebsampler → eb_gamma ∈ EB_GAMMAS

All other parameters (T, n_samples, temperature, denoise, cutoff, etc.) come
from the YAML config file passed via --config. The config's `data.l` and
`schedule.*` are IGNORED in this script — L is swept from LENGTHS below, and
the schedule is built per-cell based on the strategy.

Logging
-------
Each cell logs mean ± std (across n_reps) for EVERY metric in STATS_NAMES, plus
the mean and max number of denoising steps across all sequences in all reps.
The CSV column count is therefore 2 * len(STATS_NAMES) + a handful of
identifier and timing fields. See CSV_FIELDS construction below.

evaluation_from_generation must return 6 values:
    (stats, _, _, _, _, n_steps_per_seq)
where stats is an iterable of float metrics in the order declared by
STATS_NAMES, and n_steps_per_seq is a list of ints — one entry per generated
sequence — giving the number of unmask iterations that sequence used.

Usage
-----
  python src/oracle/eval_oracle_param_sweep.py --config configs/config_oracle.yaml --n-evals 4 --out-dir results/Dyck-sweep-param --no-resume

  # Selective re-runs:
  python src/oracle/eval_oracle_param_sweep.py --config configs/config_oracle.yaml \\
      --strategies ebsampler --gammas 0.1 0.9 --lengths 128

  # Just gaussian sigma sweep:
  python src/oracle/eval_oracle_param_sweep.py --config configs/config_oracle.yaml \\
      --strategies gaussian --lengths 128
"""

import argparse
import csv
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC  = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracle.grammar_oracles import oracleModel
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import CategoricalSchedule, GaussianSchedule


# ---------------------------------------------------------------------------
# Parameter grid  (edit these to control the sweep)
# ---------------------------------------------------------------------------
# GRAMMARS            = ["baN", "bbaN", "aNbN", "aNbNcN"]
GRAMMARS = ['not_nested_parentheses_and_brackets', 'parentheses_and_brackets']
# LENGTHS             = [128]
LENGTHS             = [32]
SAMPLING_STRATEGIES = ["greedy", "categorical"]

EB_GAMMAS       = [0.1, 0.5, 0.9, 2.0, 10.0]
# Gaussian sigma values cover ~1.5 orders of magnitude. For L=128 the
# previously-effective σ was on the order of L/10 to L/5, so this range
# brackets that and also probes both extremes (very tight / very loose).
GAUSSIAN_SIGMAS = [0.5, 2.0, 5.0, 10.0, 20.0]

# Strategy registry. Each strategy declares:
#   decoder      : decoder string passed to evaluation_from_generation
#   param_name   : column name for its hyperparameter (None if it has none)
#   param_values : list of values to sweep (use [None] for no-param strategies)
#
# NOTE: 'ar' assumes the decoder alias in your codebase is 'ar'. If it's
# 'autoregressive' instead, change the 'decoder' field below.
STRATEGIES = {
    'uniform':   {'decoder': 'schedule_driven', 'param_name': None,       'param_values': [None]},
    'gaussian':  {'decoder': 'schedule_driven', 'param_name': 'sigma',    'param_values': GAUSSIAN_SIGMAS},
    'ar':        {'decoder': 'ar',              'param_name': None,       'param_values': [None]},
    'ebsampler': {'decoder': 'ebsampler',       'param_name': 'eb_gamma', 'param_values': EB_GAMMAS},
}

# Names of the metrics returned by evaluation_from_generation, in order. The
# primary metric (the one used for the headline cell in the text table) is
# PRIMARY_STAT, which must appear in STATS_NAMES.
#
# This must match the order of the `stats` tuple/list returned by
# evaluation_from_generation. If your evaluator returns them in a different
# order or with extra / missing entries, edit this list — every other piece of
# logging is driven from it.
STATS_NAMES = [
    'rule1',
    'rule2',
    'both_rules',
    'format'
]
PRIMARY_STAT = 'both_rules'


# ---------------------------------------------------------------------------
# Strategy semantics
# ---------------------------------------------------------------------------

def is_deterministic(strategy: str, sampler: str) -> bool:
    """Cells whose output is fully determined by the starting X.

    AR and EBSampler both pick positions deterministically; greedy sampling is
    argmax. Schedule-driven (uniform / gaussian) has Bernoulli randomness in
    position selection regardless of sampler.
    """
    return sampler == 'greedy' and strategy in ('ar', 'ebsampler')


def build_schedule(strategy: str, param_value):
    """Schedule object passed to evaluation_from_generation.

    Only meaningful for schedule_driven decoders. AR / EB ignore it but the
    evaluation API still wants something non-None passed, so we hand them a
    categorical schedule as a no-op placeholder.
    """
    if strategy == 'gaussian':
        assert param_value is not None, "Gaussian strategy requires sigma."
        return GaussianSchedule(sigma=param_value)
    return CategoricalSchedule()


def build_grid():
    """List of (grammar, L, strategy, sampler, param_value) cells."""
    cells = []
    for grammar in GRAMMARS:
        for L in LENGTHS:
            for strat_name, strat in STRATEGIES.items():
                for sampler in SAMPLING_STRATEGIES:
                    for pv in strat['param_values']:
                        cells.append((grammar, L, strat_name, sampler, pv))
    return cells


def n_reps_for(strategy: str, sampler: str, n_evals_requested: int) -> int:
    return 1 if is_deterministic(strategy, sampler) else n_evals_requested


# ---------------------------------------------------------------------------
# Grammar factory
# ---------------------------------------------------------------------------

def make_grammar(grammar_name, l):
    if grammar_name == 'anbn':
        from datasets.anbn import anbnGrammar
        return anbnGrammar(l)
    from datasets.re_grammar import REGrammar
    return REGrammar(grammar_name, l)


def vocab_size_for(grammar):
    return getattr(grammar, 'vocab_size', 6)


# ---------------------------------------------------------------------------
# RNG / device
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str='auto'):
    if device_str != 'auto':
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def grammar_l_seed(seed: int, grammar: str, L: int) -> int:
    """Base seed for (grammar, L). Different grammars and different lengths
    get independent input distributions."""
    return seed + 1009 * GRAMMARS.index(grammar) + 7919 * LENGTHS.index(L)


def rep_seed(base_seed: int, rep_idx: int) -> int:
    """Per-rep seed shared across all (strategy, sampler, param) cells of a
    given (grammar, L) — matched-pair design."""
    return base_seed + rep_idx


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config_file(path):
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
        return obj

    return _ns(raw)


def cfg_get(cfg, *attrs, default=None):
    obj = cfg
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return default
    return obj


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def evaluate_cell(*, oracle, grammar, eval_dataset,
                  strategy, sampler, param_value,
                  cfg, device, temperature, T):
    """Run one evaluation pass.

    Returns
    -------
    stats : tuple of floats, length == len(STATS_NAMES)
        The full metrics vector for this rep, in the order declared by
        STATS_NAMES. NaN-tolerant: missing or undefined metrics (e.g. the
        '[Finished only] …' family when no sequence finished) propagate as NaN
        and are handled by nanmean / nanstd at aggregation time.
    n_steps_per_seq : list of int
        Number of unmask iterations used per generated sequence in this rep.
    """
    decoder    = STRATEGIES[strategy]['decoder']
    schedule   = build_schedule(strategy, param_value)
    is_gauss   = strategy == 'gaussian'
    # EB cells pass their gamma; everyone else passes a placeholder that the
    # decoder will ignore.
    gamma_pass = float(param_value) if strategy == 'ebsampler' else 0.1
    sigma_pass = float(param_value) if is_gauss else 1.0

    stats, _, _, _, _, n_steps_per_seq = evaluation_from_generation(
        oracle,
        grammar,
        evaluation_dataset=eval_dataset,
        T=T,
        decoding_strategy=decoder,
        sampling_strategy=sampler,
        temperature=temperature,
        eb_gamma=gamma_pass,
        write_steps=False,
        device=device,
        figures_path=None,
        loss_log_path=None,
        output_path=None,
        save_mode=False,
        schedule=schedule,
        gaussian_noise=is_gauss,
        sigma=sigma_pass,
        denoise=cfg_get(cfg, 'training', 'denoise', default='0'),
        cutoff=cfg_get(cfg, 'evaluation', 'cutoff', default=None),
    )
    
    print(n_steps_per_seq)

    # Coerce to a uniform shape regardless of what the evaluator returns.
    stats_tuple = tuple(float(s) for s in stats)
    if len(stats_tuple) != len(STATS_NAMES):
        raise ValueError(
            f"evaluation_from_generation returned {len(stats_tuple)} stats; "
            f"STATS_NAMES has {len(STATS_NAMES)} entries. "
            f"Update STATS_NAMES to match the evaluator's output order."
        )
    return stats_tuple, [int(n) for n in n_steps_per_seq]


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def aggregate_reps(stats_per_rep, n_steps_all):
    """Reduce a cell's reps into a single record.

    stats_per_rep : list of tuples — one tuple of floats per rep.
    n_steps_all   : flat list of ints, all sequences across all reps.
    """
    if not stats_per_rep:
        empty = {f'mean_{n}': float('nan') for n in STATS_NAMES}
        empty.update({f'std_{n}': float('nan') for n in STATS_NAMES})
        empty['n_steps_mean'] = float('nan')
        empty['n_steps_max']  = 0
        return empty

    arr = np.array(stats_per_rep, dtype=float)  # (n_reps, n_stats)
    means = np.nanmean(arr, axis=0)
    stds  = np.nanstd(arr, axis=0)

    out = {}
    for i, name in enumerate(STATS_NAMES):
        out[f'mean_{name}'] = float(means[i])
        out[f'std_{name}']  = float(stds[i])

    if n_steps_all:
        out['n_steps_mean'] = float(np.mean(n_steps_all))
        out['n_steps_max']  = int(np.max(n_steps_all))
    else:
        out['n_steps_mean'] = float('nan')
        out['n_steps_max']  = 0

    return out


# ---------------------------------------------------------------------------
# CSV I/O with resume
# ---------------------------------------------------------------------------

def _build_csv_fields():
    fields = [
        'dataset', 'grammar', 'L', 'strategy', 'sampling_strategy',
        'sigma', 'eb_gamma',
        'T', 'n_reps',
    ]
    for n in STATS_NAMES:
        fields.append(f'mean_{n}')
        fields.append(f'std_{n}')
    fields.extend(['n_steps_mean', 'n_steps_max', 'deterministic', 'elapsed_s'])
    return fields


CSV_FIELDS = _build_csv_fields()


def cell_key(grammar, L, strategy, sampler, param_value):
    pv = 0.0 if param_value is None else float(param_value)
    return (str(grammar), int(L), str(strategy), str(sampler), pv)


def _row_to_key(row):
    strategy = row['strategy']
    if strategy == 'gaussian':
        pv = row.get('sigma') or None
    elif strategy == 'ebsampler':
        pv = row.get('eb_gamma') or None
    else:
        pv = None
    return cell_key(
        row['grammar'], int(row['L']),
        row['strategy'], row['sampling_strategy'], pv,
    )


def load_completed(csv_path: Path):
    """Return {cell_key: row} for completed cells in an existing CSV."""
    completed = {}
    if not csv_path.exists():
        return completed
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed[_row_to_key(row)] = row
            except (KeyError, ValueError):
                continue
    return completed


def append_csv_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def row_label(grammar, L, strategy, sampler, param_value):
    param_name = STRATEGIES[strategy]['param_name']
    if param_name == 'sigma':
        param_str = f"σ={param_value}"
    elif param_name == 'eb_gamma':
        param_str = f"γ={param_value}"
    else:
        param_str = "—"
    return (f"{grammar:<7} L={L:<4} strat={strategy:<10} "
            f"samp={sampler:<12} {param_str}")


def make_table(rows, dataset_name):
    if not rows:
        return f"\n=== Dataset: {dataset_name} — no results ===\n"

    label_w = max(len(r['label']) for r in rows) + 2
    cell_w  = 25
    steps_w = 18

    # Create headers dynamically for every stat in STATS_NAMES
    stat_headers = [f"{stat} (mean ± std)" for stat in STATS_NAMES]
    stats_header_str = "  ".join(f"{h:<{cell_w}}" for h in stat_headers)

    header = (f"{'Config':<{label_w}}  "
              f"{stats_header_str}  "
              f"{'n_steps (mean / max)':<{steps_w}}  n_reps")
    
    # Scale separator line to the full table width
    sep = '─' * len(header)

    lines = [
        f"\n=== Dataset: {dataset_name} ===",
        header,
        sep,
    ]
    for r in rows:
        # Build individual stat cells dynamically 
        stat_cells = []
        for stat in STATS_NAMES:
            mean_val = r['stat_means'].get(stat, float('nan'))
            std_val  = r['stat_stds'].get(stat,  float('nan'))
            if r['deterministic']:
                cell = f"{mean_val:.4f} (det.)"
            else:
                cell = f"{mean_val:.4f} ± {std_val:.4f}"
            stat_cells.append(f"{cell:<{cell_w}}")
            
        stats_str = "  ".join(stat_cells)
        steps_str = f"{r['n_steps_mean']:.1f} / {r['n_steps_max']}"
        
        lines.append(
            f"{r['label']:<{label_w}}  "
            f"{stats_str}  "
            f"{steps_str:<{steps_w}}  "
            f"{r['n_reps']}"
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------

class TeeStream:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log    = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the oracle across grammar × L × strategy × sampler × "
            "per-strategy hyperparameter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config (e.g. configs/config_oracle.yaml). '
                             'data.l and schedule.* are ignored.')
    parser.add_argument('--n-evals', type=int, default=20,
                        help='Reps per stochastic cell (default 20). '
                             'Deterministic cells always run once.')
    parser.add_argument('--out-dir', type=str,
                        default='results/oracle_param_sweep')
    parser.add_argument('--grammars',   nargs='+', default=None)
    parser.add_argument('--lengths',    nargs='+', type=int, default=None)
    parser.add_argument('--strategies', nargs='+', default=None)
    parser.add_argument('--samplers',   nargs='+', default=None)
    parser.add_argument('--sigmas',     nargs='+', type=float, default=None,
                        help='Subset of sigmas (affects gaussian only).')
    parser.add_argument('--gammas',     nargs='+', type=float, default=None,
                        help='Subset of gammas (affects ebsampler only).')
    parser.add_argument('--no-resume',  action='store_true',
                        help='Ignore existing CSV and re-run every cell.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grid filtering
# ---------------------------------------------------------------------------

def _filter_grid(cells, args):
    def keep(cell):
        grammar, L, strategy, sampler, pv = cell
        if args.grammars   is not None and grammar  not in args.grammars:   return False
        if args.lengths    is not None and L        not in args.lengths:    return False
        if args.strategies is not None and strategy not in args.strategies: return False
        if args.samplers   is not None and sampler  not in args.samplers:   return False
        if strategy == 'gaussian'  and args.sigmas is not None:
            if not any(abs(pv - s) < 1e-9 for s in args.sigmas):
                return False
        if strategy == 'ebsampler' and args.gammas is not None:
            if not any(abs(pv - g) < 1e-9 for g in args.gammas):
                return False
        return True

    return [c for c in cells if keep(c)]


def _grid_summary(cells):
    counts = {}
    for _, _, strategy, _, _ in cells:
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config_file(args.config)

    device  = get_device(cfg_get(cfg, 'device', default='auto'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'run.log'
    _log_file = open(log_path, 'a', encoding='utf-8')
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(_orig_stdout, _log_file)
    sys.stderr = TeeStream(_orig_stderr, _log_file)

    print(f"\n{'#' * 60}")
    print(f"# Sweep started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")

    try:
        _run(args, cfg, device, out_dir)
    finally:
        print(f"\nLog: {log_path}")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()


def _row_from_csv(row, label):
    """Rebuild the in-memory table-row dict from a CSV record (for resume)."""
    stat_means = {n: float(row.get(f'mean_{n}', 'nan')) for n in STATS_NAMES}
    stat_stds  = {n: float(row.get(f'std_{n}',  'nan')) for n in STATS_NAMES}
    return {
        'label':         label,
        'stat_means':    stat_means,
        'stat_stds':     stat_stds,
        'n_steps_mean':  float(row.get('n_steps_mean', 'nan')),
        'n_steps_max':   int(float(row.get('n_steps_max', 0))),
        'n_reps':        int(row['n_reps']),
        'deterministic': str(row.get('deterministic')).lower() == 'true',
        'grammar':       row['grammar'],
        'L':             int(row['L']),
        'strategy':      row['strategy'],
        'sampler':       row['sampling_strategy'],
        'param':         (float(row['sigma']) if row['strategy'] == 'gaussian'
                          and row.get('sigma')
                          else float(row['eb_gamma']) if row['strategy'] == 'ebsampler'
                          and row.get('eb_gamma')
                          else None),
    }


def _run(args, cfg, device, out_dir):
    seed              = cfg_get(cfg, 'seed',                       default=2024)
    temperature       = cfg_get(cfg, 'temperature',                default=1.0)
    n_samples         = cfg_get(cfg, 'evaluation', 'n_samples',    default=500)
    eval_dataset_type = cfg_get(cfg, 'evaluation', 'eval_dataset', default='unconditional')
    eval_type         = cfg_get(cfg, 'evaluation', 'eval_type',    default='random')
    T                 = cfg_get(cfg, 'model', 'T',                 default=100)
    sampling_eps      = cfg_get(cfg, 'model', 'sampling_eps',      default=1e-5)

    print(f"\nFixed config:")
    print(f"  seed={seed}  T={T}  n_samples={n_samples}")
    print(f"  temperature={temperature}  eval_dataset={eval_dataset_type}")
    print(f"  (NOTE: data.l and schedule.* in config are ignored; this script "
          f"sweeps L from LENGTHS and selects schedule per strategy.)")

    print(f"\nSweep axes:")
    print(f"  GRAMMARS        = {GRAMMARS}")
    print(f"  LENGTHS         = {LENGTHS}")
    print(f"  STRATEGIES      = {list(STRATEGIES.keys())}")
    print(f"  SAMPLERS        = {SAMPLING_STRATEGIES}")
    print(f"  EB_GAMMAS       = {EB_GAMMAS}")
    print(f"  GAUSSIAN_SIGMAS = {GAUSSIAN_SIGMAS}")
    print(f"  STATS_NAMES     = {STATS_NAMES}  (primary = {PRIMARY_STAT})")

    full_grid = build_grid()
    cells = _filter_grid(full_grid, args)
    summary = _grid_summary(cells)
    print(f"\nGrid: {len(cells)} cells after filtering (from {len(full_grid)} total).")
    print(f"  by strategy: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))

    csv_path = out_dir / 'oracle_param_sweep_results.csv'
    completed = {} if args.no_resume else load_completed(csv_path)
    if completed and not args.no_resume:
        print(f"Resume: {len(completed)} completed cells found in {csv_path.name}.")

    cache = {}
    rows_for_table = []

    for cell_idx, (grammar_name, L, strategy, sampler, pv) in enumerate(cells, start=1):
        cell_id = f"[{cell_idx}/{len(cells)}]"
        label   = row_label(grammar_name, L, strategy, sampler, pv)
        key     = cell_key(grammar_name, L, strategy, sampler, pv)

        if key in completed:
            row = completed[key]
            print(f"\n{cell_id} SKIP (cached): {label}")
            try:
                rows_for_table.append(_row_from_csv(row, label))
            except (KeyError, ValueError):
                pass
            continue

        print(f"\n{'=' * 60}")
        print(f"{cell_id} {label}")

        cache_key = (grammar_name, L)
        if cache_key not in cache:
            base_seed = grammar_l_seed(seed, grammar_name, L)
            set_seed(base_seed)
            grammar = make_grammar(grammar_name, L)
            grammar.generate_seq()
            vs = vocab_size_for(grammar)
            oracle = oracleModel(grammar_name=grammar_name,
                                 vocab_size=vs, device=device)
            set_seed(base_seed)
            eval_ds = EvaluationDataset(
                l=L, eval_dataset=eval_dataset_type,
                eval_type=eval_type, n_samples=n_samples,
                T=T, sampling_eps=sampling_eps, device=device,
            )
            eval_ds.data = eval_ds.data.to(device)
            cache[cache_key] = {
                'grammar':   grammar,
                'oracle':    oracle,
                'eval_ds':   eval_ds,
                'base_seed': base_seed,
                'vocab':     vs,
            }
            print(f"  built ({grammar_name}, L={L})  vocab={vs}  "
                  f"|X|={eval_ds.data.shape[0]}")
        cached = cache[cache_key]

        n_reps = n_reps_for(strategy, sampler, args.n_evals)
        if is_deterministic(strategy, sampler) and args.n_evals > 1:
            print(f"  deterministic cell — running 1 rep instead of {args.n_evals}")

        # ---- run reps -------------------------------------------------------
        stats_per_rep = []   # list of tuples, length n_reps × len(STATS_NAMES)
        all_n_steps   = []   # flat list of ints, n_reps × n_samples
        t0 = time.time()
        try:
            for rep_i in range(n_reps):
                s = rep_seed(cached['base_seed'], rep_i)
                set_seed(s)
                rep_stats, rep_n_steps = evaluate_cell(
                    oracle=cached['oracle'],
                    grammar=cached['grammar'],
                    eval_dataset=cached['eval_ds'],
                    strategy=strategy,
                    sampler=sampler,
                    param_value=pv,
                    cfg=cfg,
                    device=device,
                    temperature=temperature,
                    T=T,
                )
                stats_per_rep.append(rep_stats)
                all_n_steps.extend(rep_n_steps)
                if n_reps > 1:
                    primary_idx = STATS_NAMES.index(PRIMARY_STAT)
                    print(f"  rep {rep_i + 1}/{n_reps}  seed={s}  "
                          f"{PRIMARY_STAT}={rep_stats[primary_idx]:.4f}  "
                          f"steps={np.mean(rep_n_steps):.1f}")
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  CELL FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ---- aggregate ------------------------------------------------------
        agg = aggregate_reps(stats_per_rep, all_n_steps)
        det = is_deterministic(strategy, sampler)

        primary_mean = agg[f'mean_{PRIMARY_STAT}']
        primary_std  = agg[f'std_{PRIMARY_STAT}']
        if det:
            print(f"  → {PRIMARY_STAT}={primary_mean:.4f} (deterministic)  "
                  f"steps mean/max = {agg['n_steps_mean']:.1f} / {agg['n_steps_max']}  "
                  f"elapsed={elapsed:.1f}s")
        else:
            print(f"  → {PRIMARY_STAT}={primary_mean:.4f} ± {primary_std:.4f}  "
                  f"over {n_reps} reps  "
                  f"steps mean/max = {agg['n_steps_mean']:.1f} / {agg['n_steps_max']}  "
                  f"elapsed={elapsed:.1f}s")

        # ---- persist --------------------------------------------------------
        row = {
            'dataset':           eval_dataset_type,
            'grammar':           grammar_name,
            'L':                 L,
            'strategy':          strategy,
            'sampling_strategy': sampler,
            'sigma':             float(pv) if strategy == 'gaussian'  else '',
            'eb_gamma':          float(pv) if strategy == 'ebsampler' else '',
            'T':                 T,
            'n_reps':            n_reps,
            'deterministic':     det,
            'elapsed_s':         round(elapsed, 2),
            'n_steps_mean':      round(agg['n_steps_mean'], 4),
            'n_steps_max':       agg['n_steps_max'],
        }
        for name in STATS_NAMES:
            row[f'mean_{name}'] = round(agg[f'mean_{name}'], 6)
            row[f'std_{name}']  = round(agg[f'std_{name}'],  6)

        append_csv_row(csv_path, row)

        rows_for_table.append({
            'label':         label,
            'stat_means':    {n: agg[f'mean_{n}'] for n in STATS_NAMES},
            'stat_stds':     {n: agg[f'std_{n}']  for n in STATS_NAMES},
            'n_steps_mean':  agg['n_steps_mean'],
            'n_steps_max':   agg['n_steps_max'],
            'n_reps':        n_reps,
            'deterministic': det,
            'grammar':       grammar_name, 'L': L,
            'strategy':      strategy, 'sampler': sampler, 'param': pv,
        })

    print('\n' + '=' * 60)
    print('SWEEP COMPLETE')
    print('=' * 60)

    if rows_for_table:
        rows_for_table.sort(key=lambda r: (
            r['grammar'], r['L'], r['strategy'],
            r['sampler'], 0.0 if r['param'] is None else float(r['param']),
        ))
        table = make_table(rows_for_table, eval_dataset_type)
        print(table)

        table_path = out_dir / 'oracle_param_sweep_table.txt'
        with open(table_path, 'w') as f:
            f.write(table + '\n')
        print(f"\nTable: {table_path}")
        print(f"CSV:   {csv_path}")


if __name__ == '__main__':
    main()