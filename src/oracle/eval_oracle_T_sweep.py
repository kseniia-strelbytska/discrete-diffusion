"""
eval_oracle_T_sweep.py
======================
Sweep the oracle across (grammar × L × strategy × sampler × T × hyperparam).

This extends eval_oracle_param_sweep.py by adding T as a sweep axis. The
purpose is to measure accuracy-vs-compute Pareto frontiers: for each grammar
and decoder, how does accuracy degrade as we reduce the number of forward
passes T?

Parallelism
-----------
With --workers N (default 1), cells are dispatched to a ProcessPoolExecutor.
Workers default to running on CPU (override with --worker-device, e.g. 'cuda').
Each worker process maintains its own (grammar, L, T) cache across the cells
it handles, so the build cost is amortised even with no cross-process sharing.

T semantics per strategy
------------------------
  uniform   : T = number of scheduled timesteps. Forward passes ≈ T.
  gaussian  : T = number of scheduled timesteps. Forward passes ≈ T.
  ar        : commits one position per step → forward passes ≈ L.
              We run AR ONCE per grammar at T = L + 2 (margin for EOS).
              AR is the compute-ceiling reference, not a T-curve.
  ebsampler : T = upper bound on forward passes. EB may stop early via
              mop-up. So effective compute ≤ T.

For uniform / gaussian / EB we sweep T ∈ T_VALUES.
For AR we run a single cell per (grammar, sampler) at T = L + 2.

Usage
-----
  # 28-core parallel run:
  python src/oracle/eval_oracle_T_sweep.py \
      --config configs/config_oracle.yaml \
      --out-dir results/T-sweep-nonDyck \
      --n-evals 4 --workers 28

  # Sequential (original behaviour):
  python src/oracle/eval_oracle_T_sweep.py \\
      --config configs/config_oracle.yaml --out-dir results/T-sweep-nonDyck
      
python src/oracle/eval_oracle_T_sweep.py \
      --config configs/config_oracle.yaml \
      --out-dir results/T-sweep-Dyck \
      --n-evals 5 --workers 28 

"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Sweep grid (edit these to control the sweep)
# ---------------------------------------------------------------------------
# GRAMMARS = ['baN', 'bbaN', 'aNbN', 'aNbNcN']
GRAMMARS = ['parentheses_and_brackets', 'not_nested_parentheses_and_brackets']
LENGTHS  = [32]
SAMPLING_STRATEGIES = ['greedy', 'categorical']

T_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
GAUSSIAN_SIGMAS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
EB_GAMMAS       = [0.1, 0.5, 0.9, 2.0, 5.0, 10.0]

STRATEGIES = {
    'uniform':   {'decoder': 'schedule_driven', 'param_name': None,       'param_values': [None],
                  'T_sweep': True, 'fixed_T_fn': None},
    'gaussian':  {'decoder': 'schedule_driven', 'param_name': 'sigma',    'param_values': GAUSSIAN_SIGMAS,
                  'T_sweep': True, 'fixed_T_fn': None},
    'ar':        {'decoder': 'ar',              'param_name': None,       'param_values': [None],
                  'T_sweep': False, 'fixed_T_fn': lambda L: L + 2},
    'ebsampler': {'decoder': 'ebsampler',       'param_name': 'eb_gamma', 'param_values': EB_GAMMAS,
                  'T_sweep': True, 'fixed_T_fn': None},
}

STATS_NAMES = ['rule1', 'rule2', 'both_rules', 'format']
PRIMARY_STAT = 'both_rules'

# Module-level cache of (grammar, L, T) → built state. Populated by
# _process_cell as it runs. In sequential mode, this is shared across all
# cells; in parallel mode each worker process has its own independent copy.
_worker_cache = {}


# ---------------------------------------------------------------------------
# Strategy semantics
# ---------------------------------------------------------------------------

def is_deterministic(strategy: str, sampler: str) -> bool:
    return sampler == 'greedy' and strategy in ('ar', 'ebsampler')


def build_schedule(strategy: str, param_value):
    if strategy == 'gaussian':
        assert param_value is not None, 'Gaussian strategy requires sigma.'
        return GaussianSchedule(sigma=param_value)
    return CategoricalSchedule()


def Ts_for_strategy(strategy: str, L: int):
    spec = STRATEGIES[strategy]
    if spec['T_sweep']:
        return list(T_VALUES)
    return [spec['fixed_T_fn'](L)]


def build_grid():
    cells = []
    for grammar in GRAMMARS:
        for L in LENGTHS:
            for strat_name in STRATEGIES:
                for sampler in SAMPLING_STRATEGIES:
                    for T in Ts_for_strategy(strat_name, L):
                        for pv in STRATEGIES[strat_name]['param_values']:
                            cells.append((grammar, L, strat_name, sampler, T, pv))
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
    return seed + 1009 * GRAMMARS.index(grammar) + 7919 * LENGTHS.index(L)


def rep_seed(base_seed: int, rep_idx: int) -> int:
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

def evaluate_cell(*, oracle, grammar, eval_dataset, strategy, sampler,
                  param_value, cfg, device, temperature, T):
    decoder    = STRATEGIES[strategy]['decoder']
    schedule   = build_schedule(strategy, param_value)
    is_gauss   = strategy == 'gaussian'
    gamma_pass = float(param_value) if strategy == 'ebsampler' else 0.1
    sigma_pass = float(param_value) if is_gauss else 1.0

    stats, _, _, _, _, n_steps_per_seq, correct_sequences = evaluation_from_generation(
        oracle, grammar, evaluation_dataset=eval_dataset,
        T=T,
        decoding_strategy=decoder, sampling_strategy=sampler,
        temperature=temperature, eb_gamma=gamma_pass,
        write_steps=False, device=device,
        figures_path=None, loss_log_path=None, output_path=None,
        save_mode=False, schedule=schedule,
        gaussian_noise=is_gauss, sigma=sigma_pass,
        denoise=cfg_get(cfg, 'training', 'denoise', default='0'),
        cutoff=cfg_get(cfg, 'evaluation', 'cutoff', default=None),
    )

    stats_tuple = tuple(float(s) for s in stats)
    if len(stats_tuple) != len(STATS_NAMES):
        raise ValueError(
            f'evaluation_from_generation returned {len(stats_tuple)} stats; '
            f'STATS_NAMES has {len(STATS_NAMES)} entries.'
        )
    return stats_tuple, [int(n) for n in n_steps_per_seq], correct_sequences


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n (z=1.96).

    Pooled over all generations in the cell (k correct out of n total), this
    is a more honest precision estimate than the std across the n_reps reps,
    which with only ~5 reps is itself very noisy. Returns (lo, hi); NaNs if n==0.
    """
    if n <= 0:
        return float('nan'), float('nan')
    phat   = k / n
    denom  = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half   = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return center - half, center + half


def aggregate_reps(stats_per_rep, n_steps_all):
    if not stats_per_rep:
        out = {f'mean_{n}': float('nan') for n in STATS_NAMES}
        out.update({f'std_{n}': float('nan') for n in STATS_NAMES})
        out['n_steps_mean'] = float('nan')
        out['n_steps_max']  = 0
        return out

    arr   = np.array(stats_per_rep, dtype=float)
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
# Diversity fields and helpers
# ---------------------------------------------------------------------------

DIVERSITY_FIELDS = [
    'n_correct', 'n_correct_too_low', 'uniqueness', 'duplication_rate',
    'mean_lev_dist_normalized', 'lev_n_used', 'bigram_diversity', 'trigram_diversity',
    'dfa_state_coverage', 'dfa_transition_coverage', 'n_entropy', 'n_coverage',
    'm_entropy', 'nm_joint_coverage', 'max_depth_ratio_mean', 'max_depth_ratio_std',
    'brackets_parens_ratio_mean', 'brackets_parens_ratio_std',
    'n_zero_paren_sequences', 'distribution_path',
]


def _cell_id_str(grammar_name, L, strategy, sampler, T, pv):
    if strategy == 'gaussian':
        ps = f'sigma{pv}'
    elif strategy == 'ebsampler':
        ps = f'gamma{pv}'
    else:
        ps = 'none'
    return f'{grammar_name}_L{L}_T{T}_{strategy}_{sampler}_{ps}'


def _serialise_dist(div_dist):
    """Convert a diversity_distributions() dict into picklable / JSON-able form."""
    if div_dist is None:
        return None
    out = {}
    for k, v in div_dist.items():
        if hasattr(v, 'tolist'):
            out[k] = v.tolist()
        elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _build_csv_fields():
    fields = ['dataset', 'grammar', 'L', 'strategy', 'sampling_strategy',
              'sigma', 'eb_gamma', 'T', 'n_reps']
    for n in STATS_NAMES:
        fields.append(f'mean_{n}')
        fields.append(f'std_{n}')
    fields.extend([f'n_eval_total',
                   f'ci_low_{PRIMARY_STAT}', f'ci_high_{PRIMARY_STAT}'])
    fields.extend(['n_steps_mean', 'n_steps_max', 'deterministic', 'elapsed_s'])
    fields.extend(DIVERSITY_FIELDS)
    return fields


CSV_FIELDS = _build_csv_fields()


def cell_key(grammar, L, strategy, sampler, T, param_value):
    pv = 0.0 if param_value is None else float(param_value)
    return (str(grammar), int(L), str(strategy), str(sampler), int(T), pv)


def _row_to_key(row):
    strategy = row['strategy']
    if strategy == 'gaussian':
        pv = row.get('sigma') or None
    elif strategy == 'ebsampler':
        pv = row.get('eb_gamma') or None
    else:
        pv = None
    return cell_key(row['grammar'], int(row['L']),
                    row['strategy'], row['sampling_strategy'],
                    int(row['T']), pv)


def load_completed(csv_path: Path):
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


def row_label(grammar, L, strategy, sampler, T, param_value):
    pname = STRATEGIES[strategy]['param_name']
    if pname == 'sigma':
        ps = f'σ={param_value}'
    elif pname == 'eb_gamma':
        ps = f'γ={param_value}'
    else:
        ps = '—'
    return (f'{grammar:<7} L={L:<4} T={T:<4} strat={strategy:<10} '
            f'samp={sampler:<12} {ps}')


# ---------------------------------------------------------------------------
# Per-cell work (worker function — must be picklable / importable)
# ---------------------------------------------------------------------------

def _process_cell(cell_spec, shared_cfg):
    """Run all reps for a single cell. Returns (row_dict, serial_dist, log_str).
    Pure top-level function so ProcessPoolExecutor can pickle it.

    shared_cfg keys: cfg, seed, temperature, n_samples, eval_dataset_type,
                     eval_type, sampling_eps, n_evals, device_str.
    """
    grammar_name, L, strategy, sampler, T, pv = cell_spec
    label = row_label(grammar_name, L, strategy, sampler, T, pv)
    log_buf = [f'>> {label}']

    cache_key = (grammar_name, L, T)
    if cache_key not in _worker_cache:
        base_seed = grammar_l_seed(shared_cfg['seed'], grammar_name, L)
        set_seed(base_seed)
        grammar = make_grammar(grammar_name, L)
        grammar.generate_seq()
        vs = vocab_size_for(grammar)
        device = torch.device(shared_cfg['device_str'])
        oracle = oracleModel(grammar_name=grammar_name, vocab_size=vs, device=device)
        set_seed(base_seed)
        eval_ds = EvaluationDataset(
            l=L, eval_dataset=shared_cfg['eval_dataset_type'],
            eval_type=shared_cfg['eval_type'],
            n_samples=shared_cfg['n_samples'],
            T=T, sampling_eps=shared_cfg['sampling_eps'],
            device=device,
        )
        eval_ds.data = eval_ds.data.to(device)
        _worker_cache[cache_key] = {
            'grammar': grammar, 'oracle': oracle, 'eval_ds': eval_ds,
            'base_seed': base_seed, 'device': device,
        }
        log_buf.append(f'  built ({grammar_name}, L={L}, T={T})  vocab={vs}  '
                       f'|X|={eval_ds.data.shape[0]}')
    cached = _worker_cache[cache_key]

    n_reps = n_reps_for(strategy, sampler, shared_cfg['n_evals'])

    stats_per_rep, all_n_steps, all_correct_seqs = [], [], []
    t0 = time.time()
    try:
        for rep_i in range(n_reps):
            s = rep_seed(cached['base_seed'], rep_i)
            set_seed(s)
            rep_stats, rep_n_steps, rep_correct = evaluate_cell(
                oracle=cached['oracle'], grammar=cached['grammar'],
                eval_dataset=cached['eval_ds'],
                strategy=strategy, sampler=sampler, param_value=pv,
                cfg=shared_cfg['cfg'], device=cached['device'],
                temperature=shared_cfg['temperature'], T=T,
            )
            stats_per_rep.append(rep_stats)
            all_n_steps.extend(rep_n_steps)
            all_correct_seqs.extend(rep_correct)
        elapsed = time.time() - t0
    except Exception as e:
        log_buf.append(f'  CELL FAILED: {type(e).__name__}: {e}')
        log_buf.append(traceback.format_exc())
        return None, None, '\n'.join(log_buf)

    agg = aggregate_reps(stats_per_rep, all_n_steps)
    det = is_deterministic(strategy, sampler)

    # Wilson 95% CI on the primary stat, pooled over all generations in the cell.
    n_per_rep   = cached['eval_ds'].data.shape[0]
    n_eval_total = int(n_per_rep) * len(stats_per_rep)
    k_correct    = int(round(agg[f'mean_{PRIMARY_STAT}'] * n_eval_total))
    ci_lo, ci_hi = wilson_ci(k_correct, n_eval_total)

    div_metrics, div_dist = {}, None
    if hasattr(cached['grammar'], 'diversity_metrics'):
        try:
            div_metrics = cached['grammar'].diversity_metrics(all_correct_seqs)
            div_dist = cached['grammar'].diversity_distributions(all_correct_seqs)
        except Exception as _de:
            log_buf.append(f'  diversity metrics failed: {type(_de).__name__}: {_de}')

    # Build row (distribution_path filled by main after JSON write)
    row = {
        'dataset':           shared_cfg['eval_dataset_type'],
        'grammar':           grammar_name, 'L': L, 'strategy': strategy,
        'sampling_strategy': sampler,
        'sigma':             float(pv) if strategy == 'gaussian'  else '',
        'eb_gamma':          float(pv) if strategy == 'ebsampler' else '',
        'T':                 T, 'n_reps': n_reps, 'deterministic': det,
        'elapsed_s':         round(elapsed, 2),
        'n_eval_total':      n_eval_total,
        f'ci_low_{PRIMARY_STAT}':  round(ci_lo, 6),
        f'ci_high_{PRIMARY_STAT}': round(ci_hi, 6),
        'n_steps_mean':      round(agg['n_steps_mean'], 4),
        'n_steps_max':       agg['n_steps_max'],
    }
    for name in STATS_NAMES:
        row[f'mean_{name}'] = round(agg[f'mean_{name}'], 6)
        row[f'std_{name}']  = round(agg[f'std_{name}'],  6)
    for df in DIVERSITY_FIELDS:
        if df == 'distribution_path':
            row[df] = ''
        else:
            val = div_metrics.get(df, '')
            row[df] = '' if (val != val) else val

    pmean = agg[f'mean_{PRIMARY_STAT}']
    pstd  = agg[f'std_{PRIMARY_STAT}']
    if det:
        log_buf.append(f'   → {PRIMARY_STAT}={pmean:.4f} (det)  '
                       f'steps={agg["n_steps_mean"]:.1f}/{agg["n_steps_max"]}  '
                       f'n_corr={div_metrics.get("n_correct", 0)}  '
                       f'elapsed={elapsed:.1f}s')
    else:
        log_buf.append(f'   → {PRIMARY_STAT}={pmean:.4f} ± {pstd:.4f} ({n_reps} reps)  '
                       f'steps={agg["n_steps_mean"]:.1f}/{agg["n_steps_max"]}  '
                       f'n_corr={div_metrics.get("n_correct", 0)}  '
                       f'elapsed={elapsed:.1f}s')

    return row, _serialise_dist(div_dist), '\n'.join(log_buf)


# ---------------------------------------------------------------------------
# Output writer (main-process only — no race conditions)
# ---------------------------------------------------------------------------

def _write_outputs(row, serial_dist, cell_spec, out_dir, csv_path):
    grammar_name, L, strategy, sampler, T, pv = cell_spec
    div_dist_path = ''
    if serial_dist is not None:
        dist_dir = out_dir / 'distributions'
        dist_dir.mkdir(exist_ok=True)
        cid = _cell_id_str(grammar_name, L, strategy, sampler, T, pv)
        dist_file = dist_dir / f'{cid}.json'
        with open(dist_file, 'w') as _f:
            json.dump(serial_dist, _f, indent=2)
        div_dist_path = str(dist_file)
    row['distribution_path'] = div_dist_path
    append_csv_row(csv_path, row)


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
    p = argparse.ArgumentParser(
        description='Sweep the oracle across grammar × L × strategy × sampler × T × hyperparam.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--n-evals', type=int, default=4)
    p.add_argument('--out-dir', type=str, default='results/oracle_T_sweep')
    p.add_argument('--grammars',   nargs='+', default=None)
    p.add_argument('--lengths',    nargs='+', type=int, default=None)
    p.add_argument('--strategies', nargs='+', default=None)
    p.add_argument('--samplers',   nargs='+', default=None)
    p.add_argument('--Ts',         nargs='+', type=int, default=None,
                   help='Subset of T values for swept strategies (AR ignores this).')
    p.add_argument('--sigmas',     nargs='+', type=float, default=None)
    p.add_argument('--gammas',     nargs='+', type=float, default=None)
    p.add_argument('--no-resume',  action='store_true')
    p.add_argument('--workers',    type=int, default=1,
                   help='Number of worker processes (default 1 = sequential).')
    p.add_argument('--worker-device', type=str, default='cpu',
                   help='Device for workers when --workers > 1 (default cpu).')
    return p.parse_args()


def _filter_grid(cells, args):
    def keep(cell):
        grammar, L, strategy, sampler, T, pv = cell
        if args.grammars   is not None and grammar  not in args.grammars:   return False
        if args.lengths    is not None and L        not in args.lengths:    return False
        if args.strategies is not None and strategy not in args.strategies: return False
        if args.samplers   is not None and sampler  not in args.samplers:   return False
        if args.Ts is not None and STRATEGIES[strategy]['T_sweep']:
            if T not in args.Ts: return False
        if strategy == 'gaussian'  and args.sigmas is not None:
            if not any(abs(pv - s) < 1e-9 for s in args.sigmas): return False
        if strategy == 'ebsampler' and args.gammas is not None:
            if not any(abs(pv - g) < 1e-9 for g in args.gammas): return False
        return True
    return [c for c in cells if keep(c)]


def _grid_summary(cells):
    counts = {}
    for _, _, strategy, _, _, _ in cells:
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config_file(args.config)

    main_device = get_device(cfg_get(cfg, 'device', default='auto'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'run.log'
    _log_file = open(log_path, 'a', encoding='utf-8')
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(_orig_stdout, _log_file)
    sys.stderr = TeeStream(_orig_stderr, _log_file)

    print(f'\n{"#" * 60}')
    print(f'# T-sweep started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"#" * 60}')
    print(f'Config: {args.config}')
    print(f'Output: {out_dir}')
    print(f'Workers: {args.workers}  (device: '
          f'{args.worker_device if args.workers > 1 else main_device})')

    try:
        _run(args, cfg, main_device, out_dir)
    finally:
        print(f'\nLog: {log_path}')
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()


def _run(args, cfg, main_device, out_dir):
    seed              = cfg_get(cfg, 'seed',                       default=2024)
    temperature       = cfg_get(cfg, 'temperature',                default=1.0)
    n_samples         = cfg_get(cfg, 'evaluation', 'n_samples',    default=500)
    eval_dataset_type = cfg_get(cfg, 'evaluation', 'eval_dataset', default='unconditional')
    eval_type         = cfg_get(cfg, 'evaluation', 'eval_type',    default='random')
    sampling_eps      = cfg_get(cfg, 'model', 'sampling_eps',      default=1e-5)

    device_str = args.worker_device if args.workers > 1 else str(main_device)
    shared_cfg = {
        'cfg': cfg, 'seed': seed, 'temperature': temperature,
        'n_samples': n_samples, 'eval_dataset_type': eval_dataset_type,
        'eval_type': eval_type, 'sampling_eps': sampling_eps,
        'n_evals': args.n_evals, 'device_str': device_str,
    }

    print('\nFixed config:')
    print(f'  seed={seed}  n_samples={n_samples}  device_for_cells={device_str}')
    print(f'  temperature={temperature}  eval_dataset={eval_dataset_type}')
    print('\nSweep axes:')
    print(f'  GRAMMARS        = {GRAMMARS}')
    print(f'  LENGTHS         = {LENGTHS}')
    print(f'  T_VALUES        = {T_VALUES}    (AR fixed at L+2)')
    print(f'  STRATEGIES      = {list(STRATEGIES.keys())}')
    print(f'  SAMPLERS        = {SAMPLING_STRATEGIES}')
    print(f'  EB_GAMMAS       = {EB_GAMMAS}')
    print(f'  GAUSSIAN_SIGMAS = {GAUSSIAN_SIGMAS}')

    cells = _filter_grid(build_grid(), args)
    print(f'\nGrid: {len(cells)} cells after filtering.')
    print('  by strategy: ' + ', '.join(f'{k}={v}' for k, v in sorted(_grid_summary(cells).items())))

    csv_path = out_dir / 'oracle_T_sweep_results.csv'
    completed = {} if args.no_resume else load_completed(csv_path)
    if completed:
        print(f'Resume: {len(completed)} completed cells found in {csv_path.name}.')

    cells_to_run = [c for c in cells if cell_key(*c) not in completed]
    print(f'Cells to run: {len(cells_to_run)}\n')

    if args.workers <= 1:
        # Sequential — _worker_cache used as the shared cache.
        for i, cell in enumerate(cells_to_run, start=1):
            row, serial_dist, log_str = _process_cell(cell, shared_cfg)
            print(f'[{i}/{len(cells_to_run)}] {log_str}')
            if row is not None:
                _write_outputs(row, serial_dist, cell, out_dir, csv_path)
    else:
        # Parallel — each worker maintains its own _worker_cache.
        ctx = mp.get_context('spawn')
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_process_cell, c, shared_cfg): c for c in cells_to_run}
            done = 0
            for f in as_completed(futures):
                done += 1
                cell = futures[f]
                try:
                    row, serial_dist, log_str = f.result()
                except Exception as e:
                    print(f'[{done}/{len(futures)}] WORKER CRASHED for {cell}: '
                          f'{type(e).__name__}: {e}')
                    traceback.print_exc()
                    continue
                eta = (time.time() - t_start) / done * (len(futures) - done)
                print(f'[{done}/{len(futures)}  eta={eta/60:.1f}min]\n{log_str}')
                if row is not None:
                    _write_outputs(row, serial_dist, cell, out_dir, csv_path)

    print('\n' + '=' * 60)
    print('T-SWEEP COMPLETE')
    print('=' * 60)
    print(f'CSV: {csv_path}')
    print(f'Distributions: {out_dir / "distributions"}')


if __name__ == '__main__':
    main()