"""
eval_oracle_categorical_sweep.py
================================
Evaluate the oracle model under categorical (temperature=1.0) decoding across
a 2-D grid of denoising-step counts T and Gaussian-schedule widths sigma.

Grid
----
  T      : 64, 128, 256, 512, 1024      (number of denoising steps)
  sigma  : 1, 5, 10, 20, 40, 80, 160   (Gaussian schedule boundary width)

All 5 × 7 = 35 cells use a Gaussian noise schedule; sigma controls how sharply
the masking boundary moves through the sequence.  Small sigma ≈ a sharp,
left-to-right unmasking front; large sigma ≈ soft, position-independent noise
approaching the uniform (categorical) schedule in the limit.

Why oracle + categorical?
--------------------------
The oracle has access to the exact valid-token distribution at every denoising
step, so its accuracy lower-bounds nothing — it is the theoretical ceiling for
any trained model operating under the same (T, sigma) regime.  Greedy decoding
of the oracle is fully deterministic (std = 0); categorical decoding samples
from the correct distribution and thus captures the irreducible stochastic
variance introduced by the diffusion process itself.  This experiment isolates
that variance as a function of (T, sigma) without any model approximation error.

Averaging methodology
----------------------
Each cell is evaluated N times (default 20) with different random seeds and
we report mean ± std of Rule_Accuracy/both_rules_acc.  For greedy the std
would always be 0.0; here we focus on categorical to make the N-seed averaging
informative.

Output files (all under --out-dir)
------------------------------------
  oracle_categorical_results.csv   — one row per (T, sigma) cell
  oracle_categorical_table.txt     — 2-D grid table (rows = T, columns = sigma)
  experiment_description.txt       — this experiment's metadata
  run.log                          — full terminal output captured live

Usage example
-------------
  cd /home/superuser/discrete-diffusion
  python src/eval_oracle_categorical_sweep.py

  # Custom options:
  python src/oracle/eval_oracle_categorical_sweep.py --n-samples 100 --n-evals 2 --out-dir results/oracle_categorical_T_sigma_extended --device mps
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import yaml

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oracle.grammar_oracles import oracleModel
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import GaussianSchedule

# ---------------------------------------------------------------------------
# Supported oracle grammars and grammar factory
# ---------------------------------------------------------------------------

ORACLE_GRAMMARS = [
    'anbn'
]
# , 'baN', 'bbaN', 'aNbNcN',
#   'not_nested_parentheses_and_brackets', 'parentheses_and_brackets',

def make_grammar(grammar_name, l):
    if grammar_name == 'anbn':
        from datasets.anbn import anbnGrammar
        return anbnGrammar(l)
    from datasets.re_grammar import REGrammar
    return REGrammar(grammar_name, l)


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
T_VALUES     = [128, 256]
SIGMA_VALUES = [0.2, 10, 30, 60, 80, 100]
# T \ σ           σ=1       σ=5    σ=20               σ=40               σ=160      

TEMPERATURE  = 1.0      # categorical only
DATASET_TYPE = 'unconditional'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg='auto'):
    if device_arg and device_arg != 'auto':
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _fmt_dur(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def evaluate_one(oracle, grammar, eval_dataset, T, schedule, device, cutoff):
    """Single evaluation pass. Returns both_rules_acc (float)."""
    stats, _, _, _, _ = evaluation_from_generation(
        oracle,
        grammar,
        evaluation_dataset=eval_dataset,
        T=T,
        strategy='categorical',
        temperature=TEMPERATURE,
        write_steps=False,
        device=device,
        figures_path=None,
        loss_log_path=None,
        output_path=None,
        save_mode=False,
        schedule=schedule,
        gaussian_noise=True,
        sigma=schedule.sigma,
        denoise='0',
        cutoff=cutoff,
    )
    return float(stats[2])  # both_rules_acc


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def make_2d_table(results, n_samples, n_evals, grammar_name):
    """
    Build an aligned 2-D text table.
    results: dict keyed by (T, sigma) -> (mean, std)
    Returns a multi-line string.
    """
    sigma_labels = [f"σ={s}" for s in SIGMA_VALUES]
    T_labels     = [f"T={t}" for t in T_VALUES]

    cell_w  = 17   # width of each cell
    row_lbl = max(len(l) for l in T_labels) + 2

    header_cells = '  '.join(f"{lbl:^{cell_w}}" for lbl in sigma_labels)
    header = f"{'T \\ σ':<{row_lbl}}  {header_cells}"
    sep    = '─' * len(header)

    lines = [
        f"\n=== Oracle categorical (temp={TEMPERATURE}) — both_rules_acc  "
        f"mean ± std  (n_evals={n_evals}, n_samples={n_samples}) ===",
        f"=== Grammar: {grammar_name}  Dataset: {DATASET_TYPE} ===",
        "",
        header,
        sep,
    ]

    for T in T_VALUES:
        cells = []
        for sigma in SIGMA_VALUES:
            mean, std = results[(T, sigma)]
            cells.append(f"{mean:.4f}±{std:.4f}".center(cell_w))
        lines.append(f"{f'T={T}':<{row_lbl}}  {'  '.join(cells)}")

    return '\n'.join(lines)


def save_csv(results, n_evals, out_dir):
    out_path = Path(out_dir) / 'oracle_categorical_results.csv'
    fieldnames = ['dataset', 'T', 'sigma', 'n_evals', 'mean_both_rules_acc', 'std_both_rules_acc']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for T in T_VALUES:
            for sigma in SIGMA_VALUES:
                mean, std = results[(T, sigma)]
                writer.writerow({
                    'dataset':             DATASET_TYPE,
                    'T':                   T,
                    'sigma':               sigma,
                    'n_evals':             n_evals,
                    'mean_both_rules_acc': round(mean, 6),
                    'std_both_rules_acc':  round(std, 6),
                })
    print(f"\nCSV saved to: {out_path}")


def write_description(out_dir, args, grammar_name, grammar_l, cutoff):
    path = Path(out_dir) / 'experiment_description.txt'
    lines = [
        "Experiment: Oracle model — categorical decoding — T × sigma sweep",
        "=" * 66,
        "",
        "Purpose",
        "-------",
        "Establish a theoretical upper bound on both-rules accuracy for a",
        "discrete-diffusion grammar task as a function of denoising steps T",
        "and Gaussian schedule width sigma.",
        "",
        "The oracle computes the exact valid-token probability distribution at",
        "every step, so its results are not limited by model approximation error.",
        "Comparing trained-model results (eval_sweep_checkpoints.py) against",
        "this ceiling reveals how much headroom remains.",
        "",
        "Grid",
        "----",
        f"  T      : {', '.join(str(t) for t in T_VALUES)}",
        f"  sigma  : {', '.join(str(s) for s in SIGMA_VALUES)}",
        f"  Total cells : {len(T_VALUES) * len(SIGMA_VALUES)}",
        "",
        "Decoding",
        "--------",
        f"  temperature = {TEMPERATURE}  (categorical / stochastic multinomial sampling)",
        "  Greedy (temp=0) is excluded: it is deterministic so std=0 and a",
        "  single run suffices; see eval_oracle_sweep.py for greedy results.",
        "",
        "Noise schedule",
        "--------------",
        "  All cells use a Gaussian noise schedule.",
        "  sigma controls the width of the soft masking boundary that sweeps",
        "  from right (t=0, clean) to left (t=1, fully masked).",
        "  Small sigma  → sharp front, positional masking is highly localised.",
        "  Large sigma  → diffuse front, approaches uniform (categorical) masking.",
        "",
        "Averaging",
        "---------",
        f"  {args.n_evals} independent evaluations per cell, each with a distinct seed.",
        "  We report mean ± std of both_rules_acc across those runs.",
        "  Stochastic variance from sampling the correct oracle distribution",
        "  decreases as T grows (more steps → finer corrections per token).",
        "",
        "Dataset",
        "-------",
        f"  Type   : {DATASET_TYPE}",
        f"  Samples: {args.n_samples}",
        f"  Grammar: {grammar_name}, l={grammar_l}, max_len={cutoff}",
        "",
        "Reproducibility",
        "---------------",
        "  Seeds: drawn fresh from system entropy at run start (printed to run.log).",
        "",
        "Files",
        "-----",
        "  oracle_categorical_results.csv  — one row per (T, sigma) cell",
        "  oracle_categorical_table.txt    — 2-D grid (rows=T, columns=sigma)",
        "  experiment_description.txt      — this file",
        "  run.log                         — full terminal output",
        "",
        f"Output directory: {out_dir}",
    ]
    path.write_text('\n'.join(lines) + '\n')
    print(f"Description saved to: {path}")


# ---------------------------------------------------------------------------
# Live log-file tee (identical to eval_sweep_checkpoints.py)
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
            "Oracle model, categorical decoding, sweep over T × sigma. "
            "Produces a 2-D accuracy table saved as CSV + txt + run.log."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd /home/superuser/discrete-diffusion\n"
            "  python src/oracle/eval_oracle_categorical_sweep.py \\\n"
            "      --n-samples 500 --n-evals 20 \\\n"
            "      --out-dir results/oracle_categorical_eval\n"
        ),
    )
    parser.add_argument(
        '--n-samples', type=int, default=500,
        help='Samples in the evaluation dataset (default: 500).',
    )
    parser.add_argument(
        '--n-evals', type=int, default=20,
        help='Independent evaluations per (T, sigma) cell (default: 20). Each uses a fresh random seed.',
    )
    parser.add_argument(
        '--out-dir', type=str, default='results/oracle_categorical_eval',
        help='Output directory root (default: results/oracle_categorical_eval). Results saved under <out-dir>/<grammar>/.',
    )
    parser.add_argument(
        '--grammar', type=str, default='anbn', choices=ORACLE_GRAMMARS,
        help='Grammar to evaluate (default: anbn).',
    )
    parser.add_argument(
        '--grammar-l', type=int, default=256,
        help='Max content length for the grammar (default: 256).',
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: auto, cpu, cuda, mps (default: auto).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = get_device(args.device)

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
    grammar_l = args.grammar_l
    cutoff    = grammar_l + 2

    print(f"Device: {device}")
    print(f"Grammar: {args.grammar}  l={grammar_l}  cutoff={cutoff}")
    print(f"Grid: T={T_VALUES}  sigma={SIGMA_VALUES}")
    print(f"Temperature: {TEMPERATURE} (categorical)  n_evals: {args.n_evals}  n_samples: {args.n_samples}")

    grammar = make_grammar(args.grammar, grammar_l)
    grammar.generate_seq()
    vocab_size = getattr(grammar, 'vocab_size', 6)
    oracle = oracleModel(grammar_name=args.grammar, vocab_size=vocab_size, device=device)
    print(f"vocab_size={oracle.vocab_size}")

    # Draw fresh random seeds (from system entropy) for each eval.
    # All (T, sigma) cells use the same seed list for comparability.
    eval_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.n_evals)]
    print(f"Eval seeds ({args.n_evals}): {eval_seeds}")

    print(f"\nBuilding {DATASET_TYPE} dataset ({args.n_samples} samples)...")
    eval_ds = EvaluationDataset(
        l=grammar_l,
        eval_dataset=DATASET_TYPE,
        eval_type='random',
        n_samples=args.n_samples,
        T=T_VALUES[0],
        sampling_eps=1e-5,
        device=device,
    )
    eval_ds.data = eval_ds.data.to(device)
    print(f"  actual samples: {eval_ds.data.shape[0]}")

    results = {}
    total_cells = len(T_VALUES) * len(SIGMA_VALUES)
    total_evals = total_cells * args.n_evals
    cell_idx = 0
    done_evals = 0
    start_time = time.time()
    print(f"\nTotal evaluations: {total_evals}  ({total_cells} cells × {args.n_evals} seeds)")

    for T in T_VALUES:
        for sigma in SIGMA_VALUES:
            cell_idx += 1
            schedule = GaussianSchedule(sigma=sigma)
            print(f"\n[{cell_idx}/{total_cells}] T={T}  sigma={sigma}")

            accs = []
            for eval_i in range(args.n_evals):
                cell_seed = eval_seeds[eval_i]
                set_seed(cell_seed)
                acc = evaluate_one(oracle, grammar, eval_ds, T, schedule, device, cutoff)
                accs.append(acc)
                done_evals += 1
                elapsed = time.time() - start_time
                eta = (elapsed / done_evals) * (total_evals - done_evals)
                print(f"  [{done_evals}/{total_evals} complete | {_fmt_dur(elapsed)} elapsed | ETA {_fmt_dur(eta)}]")

            mean = float(np.mean(accs))
            std  = float(np.std(accs))
            results[(T, sigma)] = (mean, std)
            print(f"  → mean={mean:.4f}  std={std:.4f}")

    # ---------------------------------------------------------------------------
    # Print and save
    # ---------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)

    table = make_2d_table(results, eval_ds.data.shape[0], args.n_evals, args.grammar)
    print(table)

    save_csv(results, args.n_evals, out_dir)

    table_path = out_dir / 'oracle_categorical_table.txt'
    table_path.write_text(table + '\n')
    print(f"Table saved to: {table_path}")

    write_description(out_dir, args, args.grammar, grammar_l, cutoff)


if __name__ == '__main__':
    main()
