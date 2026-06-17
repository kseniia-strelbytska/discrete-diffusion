"""
eval_oracle_param_sweep.py
==========================
Evaluate the oracle model across a grid of:
  - sampling_strategy : ["greedy", "categorical"]
  - eb_gamma          : [0.1, 0.5, 0.9, 2.0, 5.0, 10.0]
  - data.grammar      : ["baN", "bbaN", "aNbN", "parentheses_and_brackets",
                          "aNbNcN", "not_nested_parentheses_and_brackets"]

Fixed parameters (T, schedule, sigma, seed, grammar l, n_samples, temperature,
decoding_strategy, denoise, cutoff, …) are read from a YAML config file whose
path is passed via --config.  The file format matches config_oracle.yaml.

Only two arguments are CLI-only because they are run-level, not model-level:
  --n-evals   number of independent seed evaluations per cell (default 20)
  --out-dir   where to write CSV / table / log

Usage
-----
  cd /home/superuser/discrete-diffusion
  python src/oracle/eval_oracle_param_sweep.py --config configs/config_oracle.yaml

  # Override run-level knobs:
  python src/oracle/eval_oracle_param_sweep.py --config configs/config_oracle.yaml --n-evals 4 --out-dir results/oracle_param_sweep
"""

import argparse
import csv
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup — makes all project imports resolve regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC  = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracle.grammar_oracles import oracleModel
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import CategoricalSchedule, GaussianSchedule

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
SAMPLING_STRATEGIES = ["greedy", "categorical"]

EB_GAMMAS = [0.1, 0.5, 0.9, 2.0, 5.0, 10.0]

GRAMMARS = [
    "baN",
    "bbaN",
    "aNbN",
    "parentheses_and_brackets",
    "aNbNcN",
    "not_nested_parentheses_and_brackets",
]

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
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_schedule(schedule_type, sigma):
    if schedule_type == 'gaussian':
        return GaussianSchedule(sigma=sigma)
    return CategoricalSchedule()


def load_config_file(path):
    """Load a YAML config file and return a SimpleNamespace tree."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
        return obj

    return _ns(raw)


def cfg_get(cfg, *attrs, default=None):
    """Safe nested attribute access: cfg_get(cfg, 'schedule', 'sigma', default=1.0)."""
    obj = cfg
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return default
    return obj


def evaluate_oracle(oracle, grammar, eval_dataset, cfg, schedule, device,
                    sampling_strategy, eb_gamma, temperature=1.0):
    """Run evaluation_from_generation for one seed; returns both_rules_acc."""
    is_gaussian = isinstance(schedule, GaussianSchedule)
    stats, _, _, _, _ = evaluation_from_generation(
        oracle,
        grammar,
        evaluation_dataset=eval_dataset,
        T=cfg.model.T,
        decoding_strategy='schedule_driven',
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        eb_gamma=eb_gamma,
        write_steps=False,
        device=device,
        figures_path=None,
        loss_log_path=None,
        output_path=None,
        save_mode=False,
        schedule=schedule,
        gaussian_noise=is_gaussian,
        sigma=schedule.sigma if is_gaussian else 1.0,
        denoise=getattr(getattr(cfg, 'training', None), 'denoise', '0'),
        cutoff=getattr(getattr(cfg, 'evaluation', None), 'cutoff', None),
    )
    return float(stats[2])  # both_rules_acc


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def row_label(grammar, sampling_strategy, eb_gamma):
    return f"{grammar}  strat={sampling_strategy}  γ={eb_gamma}"


def make_table(rows, dataset_name, n_samples):
    """
    rows: list of dicts with keys:
        label, grammar, sampling_strategy, eb_gamma, mean, std
    """
    if not rows:
        return f"\n=== Dataset: {dataset_name} — no results ===\n"

    label_w = max(len(r['label']) for r in rows) + 2
    col_w   = 22

    header = (
        f"{'Config':<{label_w}}  "
        f"{'both_rules_acc (mean ± std)':<{col_w}}"
    )
    sep = '─' * (label_w + col_w + 4)

    lines = [
        f"\n=== Dataset: {dataset_name}  (n_samples={n_samples}) ===",
        header,
        sep,
    ]
    for r in rows:
        cell = f"{r['mean']:.4f} ± {r['std']:.4f}"
        lines.append(f"{r['label']:<{label_w}}  {cell:<{col_w}}")
    return '\n'.join(lines)


def save_csv(all_results, out_dir):
    out_path = Path(out_dir) / 'oracle_param_sweep_results.csv'
    fieldnames = [
        'dataset', 'grammar', 'sampling_strategy', 'eb_gamma',
        'T', 'schedule', 'sigma',
        'n_evals', 'mean_both_rules_acc', 'std_both_rules_acc',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Live log-file tee
# ---------------------------------------------------------------------------

class TeeStream:
    """Mirrors every write to both a stream and a log file."""
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
            "Evaluate the oracle model across a grid of "
            "sampling_strategy × eb_gamma × grammar, "
            "with fixed parameters read from a YAML config file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd /home/superuser/discrete-diffusion\n"
            "  python src/oracle/eval_oracle_param_sweep.py \\\n"
            "      --config configs/config_oracle.yaml \\\n"
            "      --n-evals 20 \\\n"
            "      --out-dir results/oracle_param_sweep\n"
        ),
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file (e.g. configs/config_oracle.yaml).',
    )
    # Run-level knobs only — everything else comes from the config file.
    parser.add_argument(
        '--n-evals', type=int, default=20,
        help=(
            'Number of independent seed evaluations to average per cell '
            '(default: 20). Greedy is deterministic so std will be ~0; '
            'categorical variance is characterised by these N runs.'
        ),
    )
    parser.add_argument(
        '--out-dir', type=str, default='results/oracle_param_sweep',
        help='Output directory (default: results/oracle_param_sweep).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = load_config_file(args.config)

    device_str = cfg_get(cfg, 'device', default='auto')
    if device_str == 'auto':
        device = get_device()
    else:
        device = torch.device(device_str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path  = out_dir / 'run.log'
    _log_file = open(log_path, 'w', encoding='utf-8')
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(_orig_stdout, _log_file)
    sys.stderr = TeeStream(_orig_stderr, _log_file)

    print(f"Config: {args.config}")

    try:
        _run(args, cfg, device, out_dir)
    finally:
        print(f"\nLog saved to: {log_path}")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()


def _run(args, cfg, device, out_dir):
    # -------------------------------------------------------------------------
    # Unpack fixed parameters from config
    # -------------------------------------------------------------------------
    seed            = cfg_get(cfg, 'seed',                          default=2024)
    temperature     = cfg_get(cfg, 'temperature',                   default=1.0)
    decoding_strat  = cfg_get(cfg, 'decoding_strategy',             default='schedule_driven')
    grammar_l       = cfg_get(cfg, 'data', 'l',                     default=256)
    n_samples       = cfg_get(cfg, 'evaluation', 'n_samples',       default=500)
    eval_dataset    = cfg_get(cfg, 'evaluation', 'eval_dataset',    default='unconditional')
    eval_type       = cfg_get(cfg, 'evaluation', 'eval_type',       default='random')
    cutoff          = cfg_get(cfg, 'evaluation', 'cutoff',          default=None)
    T               = cfg_get(cfg, 'model', 'T',                    default=100)
    sampling_eps    = cfg_get(cfg, 'model', 'sampling_eps',         default=1e-5)
    denoise         = cfg_get(cfg, 'training', 'denoise',           default='0')
    schedule_type   = cfg_get(cfg, 'schedule', 'type',              default='categorical')
    sigma           = cfg_get(cfg, 'schedule', 'sigma',             default=1.0)

    dataset_types = [eval_dataset]

    print(f"Device: {device}")
    print(f"Oracle model — no training required.")
    print(f"Seed: {seed}  T: {T}  Schedule: {schedule_type}  sigma: {sigma}")
    print(f"Temperature: {temperature}  Decoding: {decoding_strat}  Denoise: {denoise}")
    print(f"Grammar l: {grammar_l}  n_samples: {n_samples}  cutoff: {cutoff}")
    print(f"Grid: {len(GRAMMARS)} grammars × "
          f"{len(SAMPLING_STRATEGIES)} strategies × "
          f"{len(EB_GAMMAS)} eb_gammas = "
          f"{len(GRAMMARS) * len(SAMPLING_STRATEGIES) * len(EB_GAMMAS)} cells")

    schedule = make_schedule(schedule_type, sigma)

    all_csv_rows     = []
    per_dataset_rows = {ds: [] for ds in dataset_types}

    grid    = list(itertools.product(GRAMMARS, SAMPLING_STRATEGIES, EB_GAMMAS))
    n_cells = len(grid)

    for cell_idx, (grammar_name, sampling_strategy, eb_gamma) in enumerate(grid, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{cell_idx}/{n_cells}]  grammar={grammar_name}  "
              f"strategy={sampling_strategy}  eb_gamma={eb_gamma}")

        # Build grammar and oracle for this grammar
        set_seed(seed)
        grammar = make_grammar(grammar_name, grammar_l)
        grammar.generate_seq()
        vs      = vocab_size_for(grammar)
        oracle  = oracleModel(grammar_name=grammar_name, vocab_size=vs, device=device)
        print(f"  vocab_size={oracle.vocab_size}")

        # Unique seed base per cell (stable across reruns)
        grammar_idx  = GRAMMARS.index(grammar_name)
        strategy_idx = SAMPLING_STRATEGIES.index(sampling_strategy)
        gamma_idx    = EB_GAMMAS.index(eb_gamma)
        cell_seed_base = (
            seed
            + grammar_idx  * 10_000_000
            + strategy_idx *  1_000_000
            + gamma_idx    *    100_000
        )

        for ds_type in dataset_types:
            set_seed(seed)
            eval_ds = EvaluationDataset(
                l=grammar_l,
                eval_dataset=ds_type,
                eval_type=eval_type,
                n_samples=n_samples,
                T=T,
                sampling_eps=sampling_eps,
                device=device,
            )
            eval_ds.data = eval_ds.data.to(device)
            actual_n = eval_ds.data.shape[0]

            accs = []
            for eval_i in range(args.n_evals):
                eval_seed = cell_seed_base + eval_i
                set_seed(eval_seed)
                acc = evaluate_oracle(
                    oracle, grammar, eval_ds, cfg, schedule, device,
                    sampling_strategy=sampling_strategy,
                    eb_gamma=eb_gamma,
                    temperature=temperature,
                )
                accs.append(acc)
                print(f"  [{ds_type}] eval {eval_i + 1}/{args.n_evals}  "
                      f"seed={eval_seed}  both_rules_acc={acc:.4f}")

            mean  = float(np.mean(accs))
            std   = float(np.std(accs))
            label = row_label(grammar_name, sampling_strategy, eb_gamma)

            all_csv_rows.append({
                'dataset':             ds_type,
                'grammar':             grammar_name,
                'sampling_strategy':   sampling_strategy,
                'eb_gamma':            eb_gamma,
                'T':                   T,
                'schedule':            schedule_type,
                'sigma':               sigma if schedule_type == 'gaussian' else '',
                'n_evals':             args.n_evals,
                'mean_both_rules_acc': round(mean, 6),
                'std_both_rules_acc':  round(std,  6),
            })
            per_dataset_rows[ds_type].append({
                'label':             label,
                'grammar':           grammar_name,
                'sampling_strategy': sampling_strategy,
                'eb_gamma':          eb_gamma,
                'mean':              mean,
                'std':               std,
            })

    # -------------------------------------------------------------------------
    # Print and save results
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('ORACLE PARAM SWEEP RESULTS')
    print('=' * 60)

    table_lines = []
    for ds_type in dataset_types:
        rows = sorted(
            per_dataset_rows[ds_type],
            key=lambda r: (r['grammar'], r['sampling_strategy'], r['eb_gamma']),
        )
        set_seed(seed)
        _ds = EvaluationDataset(
            l=grammar_l, eval_dataset=ds_type, eval_type=eval_type,
            n_samples=n_samples, T=T, sampling_eps=sampling_eps, device=device,
        )
        actual_n = _ds.data.shape[0]
        table = make_table(rows, ds_type, actual_n)
        print(table)
        table_lines.append(table)

    if all_csv_rows:
        save_csv(all_csv_rows, out_dir)

    table_path = out_dir / 'oracle_param_sweep_table.txt'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines) + '\n')
    print(f"Table saved to: {table_path}")


if __name__ == '__main__':
    main()
