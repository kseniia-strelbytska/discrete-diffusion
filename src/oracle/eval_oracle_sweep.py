"""
eval_oracle_sweep.py
====================
Evaluate the oracle model under greedy vs categorical decoding for each
sweep's (T, schedule) configuration, producing the same result table as
eval_sweep_checkpoints.py — but without any trained model.

The oracle model (deterministic_token_distribution.oracleModel) requires no
training.  For any partially-masked sequence it computes the exact valid-token
distribution dictated by the a^n b^n grammar and uses that as logits.
Evaluating it under each sweep's noise schedule and T gives a theoretical
upper-bound reference: the best accuracy a diffusion model trained on that
sweep could ever achieve.

Why N independent evaluations?
-------------------------------
With greedy decoding (temperature=0) the oracle is fully deterministic: every
run with the same input produces the same output, so std=0.  With categorical
decoding (temperature=1.0) the oracle samples *from the correct distribution*,
so there is genuine stochastic variance.  Running N evaluations (default 20)
with different seeds gives a mean ± std that characterises this sampling
noise — the same averaging philosophy as using the last-20 checkpoints in
eval_sweep_checkpoints.py.

Relationship to eval_sweep_checkpoints.py
------------------------------------------
The two scripts produce identically formatted tables and CSVs so results can
be directly compared side-by-side.  The oracle row for a given sweep is the
ceiling; the trained-model row is how close that sweep's model got.

Usage example
-------------
  cd /home/superuser/discrete-diffusion
  python src/eval_oracle_sweep.py

  # Custom options:
  python src/oracle/eval_oracle_sweep.py --n-samples 100 --n-evals 2 --seed 2024 --out-dir ./results/oracle_eval
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import yaml

# Add src/ to path so all project imports resolve regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracle.grammar_oracles import oracleModel
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import CategoricalSchedule, GaussianSchedule

# ---------------------------------------------------------------------------
# Supported oracle grammars and grammar factory
# ---------------------------------------------------------------------------

# ORACLE_GRAMMARS = [
#     'anbn', 'baN', 'bbaN', 'aNbNcN',
#     'not_nested_parentheses_and_brackets', 'parentheses_and_brackets',
# ]
ORACLE_GRAMMARS = [
    'anbn'
]


def make_grammar(grammar_name, l):
    if grammar_name == 'anbn':
        from datasets.anbn import anbnGrammar
        return anbnGrammar(l)
    from datasets.re_grammar import REGrammar
    return REGrammar(grammar_name, l)


# ---------------------------------------------------------------------------
# Sweep config directories — read T and schedule from each run's saved config
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sweep config directories — read T and schedule from each run's saved config
# ---------------------------------------------------------------------------
# Using _SRC dynamically adapts to your actual project location
SWEEP_DIRS = [
    str(_ROOT / "models/sweep1-RPE-uniform-T100_12062026_214734"),
    str(_ROOT / "models/sweep2-RPE-gaussian-s20-T100_12062026_231649"),
    str(_ROOT / "models/sweep3-RPE-gaussian-s10-T100_13062026_005150"),
    str(_ROOT / "models/sweep4-RPE-gaussian-s5-T100_13062026_022706"),
    str(_ROOT / "models/sweep5-RPE-uniform-T500_13062026_040137"),
]

DATASET_TYPES = ['unconditional']

MODES = {
    'greedy':      ('greedy',      1.0),  # (sampling_strategy, temperature for logit scaling)
    'categorical': ('categorical', 1.0),
}

# ---------------------------------------------------------------------------
# Helpers (mirror eval_sweep_checkpoints.py)
# ---------------------------------------------------------------------------

def dict_to_ns(d):
    return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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


def get_schedule(cfg):
    schedule_cfg = getattr(cfg, 'schedule', None)
    schedule_type = getattr(schedule_cfg, 'type', 'categorical')
    sigma = getattr(schedule_cfg, 'sigma', 1.0)
    if schedule_type == 'gaussian':
        return GaussianSchedule(sigma=sigma)
    return CategoricalSchedule()


def sweep_label(cfg):
    schedule_type = getattr(getattr(cfg, 'schedule', None), 'type', 'categorical')
    T = cfg.model.T
    if schedule_type == 'categorical':
        return f"uniform T={T}"
    sigma = getattr(cfg.schedule, 'sigma', '?')
    return f"gaussian σ={sigma} T={T}"


def evaluate_oracle(oracle, grammar, eval_dataset, cfg, schedule, device, sampling_strategy, temperature=1.0):
    """Run evaluation_from_generation for one seed. Returns both_rules_acc."""
    is_gaussian = isinstance(schedule, GaussianSchedule)
    stats, _, _, _, _ = evaluation_from_generation(
        oracle,
        grammar,
        evaluation_dataset=eval_dataset,
        T=cfg.model.T,
        decoding_strategy='schedule_driven',
        sampling_strategy=sampling_strategy,
        temperature=temperature,
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
# Output formatting (identical layout to eval_sweep_checkpoints.py)
# ---------------------------------------------------------------------------

def make_table(rows, dataset_name, n_samples):
    if not rows:
        return f"\n=== Dataset: {dataset_name} — no results ===\n"

    label_w = max(len(r['label']) for r in rows) + 2
    col_w = 22

    header = (
        f"{'Sweep':<{label_w}}  "
        f"{'Greedy (temp=0)':<{col_w}}  "
        f"{'Categorical (temp=1)':<{col_w}}  "
        f"Gap (G−C)"
    )
    sep = '─' * len(header)

    lines = [
        f"\n=== Dataset: {dataset_name}  (n_samples={n_samples}) ===",
        header,
        sep,
    ]
    for r in rows:
        g_cell = f"{r['greedy_mean']:.4f} ± {r['greedy_std']:.4f}"
        c_cell = f"{r['cat_mean']:.4f} ± {r['cat_std']:.4f}"
        gap = r['greedy_mean'] - r['cat_mean']
        sign = '+' if gap >= 0 else ''
        lines.append(
            f"{r['label']:<{label_w}}  {g_cell:<{col_w}}  {c_cell:<{col_w}}  {sign}{gap:.4f}"
        )
    return '\n'.join(lines)


def save_csv(all_results, out_dir):
    out_path = Path(out_dir) / 'oracle_eval_results.csv'
    fieldnames = [
        'dataset', 'sweep_label', 'schedule', 'sigma', 'T',
        'n_evals',
        'greedy_mean', 'greedy_std',
        'categorical_mean', 'categorical_std',
        'gap_greedy_minus_categorical',
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
    """Mirrors every write to both a stream and a log file, flushing after each write."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

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
            "Evaluate the oracle model (no training needed) under greedy vs "
            "categorical decoding for each sweep's (T, schedule) config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd /home/superuser/discrete-diffusion\n"
            "  python src/eval_oracle_sweep.py \\\n"
            "      --n-samples 500 --n-evals 20 --seed 2024 \\\n"
            "      --out-dir results/oracle_eval\n"
        ),
    )
    parser.add_argument(
        '--n-samples', type=int, default=500,
        help='Samples per evaluation dataset (default: 500).',
    )
    parser.add_argument(
        '--n-evals', type=int, default=20,
        help=(
            'Number of independent seed evaluations to average per '
            '(sweep, mode) cell (default: 20). '
            'Greedy is deterministic so std will be 0; '
            'categorical variance is characterised by these N runs.'
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=2024,
        help='Base random seed (default: 2024). Per-eval seeds are derived from this.',
    )
    parser.add_argument(
        '--out-dir', type=str, default='results/oracle_eval',
        help='Output directory root (default: results/oracle_eval). Results are saved under <out-dir>/<grammar>/.',
    )
    parser.add_argument(
        '--grammar', type=str, default='anbn', choices=ORACLE_GRAMMARS,
        help='Grammar to evaluate (default: anbn).',
    )
    parser.add_argument(
        '--grammar-l', type=int, default=256,
        help='Max content length for the grammar (default: 256).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = get_device()

    out_dir = Path(args.out_dir) / args.grammar
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'run.log'
    _log_file = open(log_path, 'w', encoding='utf-8')
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(_orig_stdout, _log_file)
    sys.stderr = TeeStream(_orig_stderr, _log_file)

    try:
        _run(args, device, out_dir)
    finally:
        print(f"\nLog saved to: {log_path}")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()


def _run(args, device, out_dir):
    print(f"Device: {device}")
    print(f"Oracle model — no training required.")

    set_seed(args.seed)
    grammar = make_grammar(args.grammar, args.grammar_l)
    grammar.generate_seq()
    vocab_size = getattr(grammar, 'vocab_size', 6)
    oracle = oracleModel(grammar_name=args.grammar, vocab_size=vocab_size, device=device)
    print(f"Grammar: {args.grammar}  l={args.grammar_l}  vocab_size={oracle.vocab_size}")

    print("\nBuilding evaluation datasets...")
    eval_datasets = {}
    for ds_type in DATASET_TYPES:
        set_seed(args.seed)
        ds = EvaluationDataset(
            l=args.grammar_l,
            eval_dataset=ds_type,
            eval_type='random',
            n_samples=args.n_samples,
            T=100,
            sampling_eps=1e-5,
            device=device,
        )
        ds.data = ds.data.to(device)
        eval_datasets[ds_type] = ds
        print(f"  {ds_type}: {ds.data.shape[0]} samples")

    all_csv_rows = []
    per_dataset_table_rows = {ds: [] for ds in DATASET_TYPES}

    for sweep_idx, sweep_dir_str in enumerate(SWEEP_DIRS):
        sweep_dir = Path(sweep_dir_str)
        sweep_name = f"sweep{sweep_idx + 1}"

        print(f"\n{'=' * 60}")
        print(f"[{sweep_name}] {sweep_dir.name}")

        if not sweep_dir.exists():
            print(f"  [WARNING] Directory not found: {sweep_dir}. Skipping.")
            continue

        config_path = sweep_dir / 'config.yaml'
        if not config_path.exists():
            print(f"  [WARNING] config.yaml missing in {sweep_dir}. Skipping.")
            continue

        cfg = dict_to_ns(load_config(config_path))
        label = sweep_label(cfg)
        schedule = get_schedule(cfg)
        print(f"  T={cfg.model.T}, label={label!r}")

        for ds_type in DATASET_TYPES:
            eval_ds = eval_datasets[ds_type]
            mode_accs = {mode: [] for mode in MODES}

            for eval_i in range(args.n_evals):
                # Seed: unique per (sweep, dataset, eval_index)
                # Mode offset keeps greedy/categorical seeds distinct within the same eval_i.
                eval_seed = args.seed + sweep_idx * 100_000 + eval_i * 1_000
                print(f"  [{ds_type}] eval {eval_i + 1}/{args.n_evals}  seed={eval_seed}")

                for mode_name, (sampling_strategy, temperature) in MODES.items():
                    mode_offset = list(MODES).index(mode_name)
                    set_seed(eval_seed + mode_offset)
                    acc = evaluate_oracle(
                        oracle, grammar, eval_ds, cfg, schedule, device, sampling_strategy, temperature
                    )
                    mode_accs[mode_name].append(acc)
                    print(f"    {mode_name}: both_rules_acc={acc:.4f}")

            greedy_accs = mode_accs['greedy']
            cat_accs    = mode_accs['categorical']
            greedy_mean = float(np.mean(greedy_accs))
            greedy_std  = float(np.std(greedy_accs))
            cat_mean    = float(np.mean(cat_accs))
            cat_std     = float(np.std(cat_accs))
            gap         = greedy_mean - cat_mean

            all_csv_rows.append({
                'dataset':                      ds_type,
                'sweep_label':                  label,
                'schedule':                     getattr(getattr(cfg, 'schedule', None), 'type', 'categorical'),
                'sigma':                        getattr(getattr(cfg, 'schedule', None), 'sigma', ''),
                'T':                            cfg.model.T,
                'n_evals':                      args.n_evals,
                'greedy_mean':                  round(greedy_mean, 6),
                'greedy_std':                   round(greedy_std, 6),
                'categorical_mean':             round(cat_mean, 6),
                'categorical_std':              round(cat_std, 6),
                'gap_greedy_minus_categorical': round(gap, 6),
            })
            per_dataset_table_rows[ds_type].append({
                'label':       label,
                'greedy_mean': greedy_mean, 'greedy_std': greedy_std,
                'cat_mean':    cat_mean,    'cat_std':    cat_std,
            })

    # ---------------------------------------------------------------------------
    # Print and save results
    # ---------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('ORACLE RESULTS')
    print('=' * 60)

    table_lines = []
    for ds_type in DATASET_TYPES:
        rows = per_dataset_table_rows[ds_type]
        actual_n = eval_datasets[ds_type].data.shape[0]
        table = make_table(rows, ds_type, actual_n)
        print(table)
        table_lines.append(table)

    if all_csv_rows:
        save_csv(all_csv_rows, out_dir)

    table_path = out_dir / 'oracle_eval_table.txt'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines) + '\n')
    print(f"Table saved to: {table_path}")


if __name__ == '__main__':
    main()
