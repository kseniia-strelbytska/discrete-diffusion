"""
eval_sweep_checkpoints.py
=========================
Compare five trained sweep checkpoints under greedy vs categorical decoding,
with results noise-averaged across the last N checkpoints per sweep.

Methodology
-----------
Training curves for discrete-diffusion models are noisy: a single checkpoint
may be a lucky or unlucky draw from a high-variance eval curve.  Averaging
Rule_Accuracy/both_rules_acc over the *last N checkpoints* (default N=20,
corresponding to epochs 5 000 – 100 000 in 5 000-epoch increments) gives a
more stable estimate of end-of-training performance without cherry-picking a
single "best" run — the equivalent of a running-average early-stopping
estimate.  We report mean ± std across the N checkpoints so the table also
shows how stable each sweep/mode combination is.

The two decoding modes are:
  greedy      temperature=0   — deterministic argmax at every denoising step
  categorical temperature=1.0 — stochastic multinomial sampling

Each (sweep, dataset, mode, checkpoint) cell is evaluated with a fixed seed
derived from (base_seed, sweep_index, checkpoint_epoch) for reproducibility.

Datasets evaluated
------------------
  unconditional — 500 fully-masked prompts (model generates a^n b^n from scratch)
  complete      — up to n_samples partial completions from the grammar

Both datasets are created once and reused across all sweeps.

Usage example
-------------
  cd /home/superuser/discrete-diffusion
  python src/eval_sweep_checkpoints.py

  # Custom options:
  python src/eval_sweep_checkpoints.py \\
      --n-samples 500 --n-checkpoints 20 --seed 2024 \\
      --out-dir results/sweep_eval
"""

import argparse
import csv
import os
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
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import CategoricalSchedule, GaussianSchedule
from model_RPE import RPETransformerClassifier
from model import TransformerClassifier
from model_v2 import v2TransformerClassifier
from model_RPE_KQ import RPEKQTransformerClassifier
from model_FIRE import FIRETransformerClassifier
from model_T5 import T5RPETransformerClassifier
from model_timestep import TimestepTransformerClassifier

# ---------------------------------------------------------------------------
# Sweep checkpoint directories (absolute paths)
# ---------------------------------------------------------------------------
SWEEP_DIRS = [
    "/home/superuser/discrete-diffusion/models/sweep1-RPE-uniform-T100_12062026_214734",
    "/home/superuser/discrete-diffusion/models/sweep2-RPE-gaussian-s20-T100_12062026_231649",
    "/home/superuser/discrete-diffusion/models/sweep3-RPE-gaussian-s10-T100_13062026_005150",
    "/home/superuser/discrete-diffusion/models/sweep4-RPE-gaussian-s5-T100_13062026_022706",
    "/home/superuser/discrete-diffusion/models/sweep5-RPE-uniform-T500_13062026_040137",
]

DATASET_TYPES = ['unconditional']

# Temperature values for each decoding mode
MODES = {
    'greedy':      0,
    'categorical': 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
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
    """Build a NoiseSchedule from the run's saved config."""
    schedule_cfg = getattr(cfg, 'schedule', None)
    schedule_type = getattr(schedule_cfg, 'type', 'categorical')
    sigma = getattr(schedule_cfg, 'sigma', 1.0)
    if schedule_type == 'gaussian':
        return GaussianSchedule(sigma=sigma)
    return CategoricalSchedule()


def build_model(cfg, device):
    """Instantiate the model architecture described in cfg (no weights loaded)."""
    arch = cfg.model.architecture
    kwargs = dict(
        max_len=cfg.model.max_len,
        vocab_size=cfg.model.vocab_size,
        n_head=cfg.model.n_head,
        n_layers=cfg.model.n_layers,
        embed_dim=cfg.model.embed_dim,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        layer_norm_eps=cfg.model.layer_norm_eps,
        sampling_eps=cfg.model.sampling_eps,
    )
    if arch == 'RPE':
        return RPETransformerClassifier(**kwargs).to(device)
    if arch == 'classic':
        return TransformerClassifier(**kwargs).to(device)
    if arch == 'v2':
        return v2TransformerClassifier(**kwargs).to(device)
    if arch == 'RPE_KQ':
        return RPEKQTransformerClassifier(**kwargs).to(device)
    if arch == 'FIRE':
        return FIRETransformerClassifier(**kwargs).to(device)
    if arch == 'T5':
        return T5RPETransformerClassifier(
            **kwargs, num_buckets=cfg.model.num_buckets
        ).to(device)
    if arch == 'timestep':
        return TimestepTransformerClassifier(**kwargs).to(device)
    raise ValueError(f"Unsupported architecture: {arch!r}")


def sweep_label(cfg):
    """Human-readable label: schedule type, sigma (if gaussian), and T."""
    schedule_type = getattr(getattr(cfg, 'schedule', None), 'type', 'categorical')
    T = cfg.model.T
    if schedule_type == 'categorical':
        return f"uniform T={T}"
    sigma = getattr(cfg.schedule, 'sigma', '?')
    return f"gaussian σ={sigma} T={T}"


def get_checkpoints(sweep_dir, n_checkpoints):
    """
    Glob model_epochs=* files, sort by epoch number (ascending),
    return the last n_checkpoints. Warns if fewer exist.
    """
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        print(f"  [WARNING] Sweep dir not found: {sweep_dir}")
        return []

    ckpts = sorted(
        [p for p in sweep_dir.glob('model_epochs=*')],
        key=lambda p: int(p.name.split('=')[1])
    )

    if not ckpts:
        print(f"  [WARNING] No checkpoints found in {sweep_dir}")
        return []

    if len(ckpts) < n_checkpoints:
        print(
            f"  [WARNING] Only {len(ckpts)} checkpoints available in "
            f"{sweep_dir.name} (requested {n_checkpoints}). Using all."
        )

    selected = ckpts[-n_checkpoints:]
    epochs = ', '.join(p.name.split('=')[1] for p in selected)
    print(f"  Selected {len(selected)} checkpoints — epochs: {epochs}")
    return selected


def load_state_dict(path, device):
    """Load a state-dict, compatible with both old and new PyTorch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only not supported in older PyTorch
        return torch.load(path, map_location=device)


def evaluate_checkpoint(model, grammar, eval_dataset, cfg, schedule, device, temperature):
    """
    Run evaluation_from_generation for one (model, dataset, temperature) combination.
    Returns both_rules_acc (float in [0, 1]).
    """
    is_gaussian = isinstance(schedule, GaussianSchedule)
    stats, _, _, _, _ = evaluation_from_generation(
        model,
        grammar,
        evaluation_dataset=eval_dataset,
        T=cfg.model.T,
        # strategy param is accepted by the function but never used by the sampler;
        # temperature alone controls greedy (<=0) vs stochastic (>0).
        strategy='categorical',
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
# Output formatting
# ---------------------------------------------------------------------------

def make_table(rows, dataset_name, n_samples):
    """Return an aligned text table string for one dataset."""
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
    out_path = Path(out_dir) / 'sweep_eval_results.csv'
    fieldnames = [
        'dataset', 'sweep_label', 'schedule', 'sigma', 'T',
        'n_checkpoints_used', 'epochs_evaluated',
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 5 sweep checkpoints under greedy vs categorical decoding, "
            "noise-averaged over the last N checkpoints per sweep."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cd /home/superuser/discrete-diffusion\n"
            "  python src/eval_sweep_checkpoints.py \\\n"
            "      --n-samples 500 --n-checkpoints 20 --seed 2024 \\\n"
            "      --out-dir results/sweep_eval\n"
        ),
    )
    parser.add_argument(
        '--n-samples', type=int, default=500,
        help='Samples per evaluation dataset (default: 500). '
             'For unconditional the dataset is always exactly 500.',
    )
    parser.add_argument(
        '--n-checkpoints', type=int, default=20,
        help='Number of last checkpoints to average per sweep (default: 20).',
    )
    parser.add_argument(
        '--seed', type=int, default=2024,
        help='Base random seed (default: 2024). Per-checkpoint seeds are derived from this.',
    )
    parser.add_argument(
        '--out-dir', type=str, default='results/sweep_eval',
        help='Output directory for CSV and table text file (default: results/sweep_eval).',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grammar is shared across all sweeps (same l=256)
    set_seed(args.seed)
    grammar = anbnGrammar(256)
    grammar.generate_seq()

    # Build evaluation datasets once; they are reused across sweeps.
    # Seeded before construction so the random subsampling is reproducible.
    print("\nBuilding evaluation datasets...")
    eval_datasets = {}
    for ds_type in DATASET_TYPES:
        set_seed(args.seed)
        ds = EvaluationDataset(
            l=256,
            eval_dataset=ds_type,
            eval_type='random',
            n_samples=args.n_samples,
            # T is only needed for the 'diffusion' dataset type; ignored here.
            T=100,
            sampling_eps=1e-5,
            device=device,
        )
        ds.data = ds.data.to(device)
        actual_n = ds.data.shape[0]
        eval_datasets[ds_type] = ds
        print(f"  {ds_type}: {actual_n} samples")

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

        # Load run's own config so T, schedule, and sigma are correct for this sweep.
        cfg = dict_to_ns(load_config(config_path))
        label = sweep_label(cfg)
        schedule = get_schedule(cfg)
        print(f"  arch={cfg.model.architecture}, T={cfg.model.T}, label={label!r}")

        checkpoints = get_checkpoints(sweep_dir, args.n_checkpoints)
        if not checkpoints:
            continue

        # Build model shell once; weights are swapped per checkpoint.
        model = build_model(cfg, device)

        for ds_type in DATASET_TYPES:
            eval_ds = eval_datasets[ds_type]
            mode_accs = {mode: [] for mode in MODES}

            for ckpt_path in checkpoints:
                epoch = int(ckpt_path.name.split('=')[1])
                # Per-checkpoint seed: unique across sweeps and epochs.
                ckpt_seed = args.seed + sweep_idx * 100_000 + epoch
                print(f"  [{ds_type}] {ckpt_path.name}  seed={ckpt_seed}")

                state_dict = load_state_dict(ckpt_path, device)
                model.load_state_dict(state_dict)

                for mode_name, temperature in MODES.items():
                    # Mode offset (0 or 1) keeps greedy and categorical seeds distinct
                    # even when they share the same checkpoint.
                    mode_offset = list(MODES).index(mode_name)
                    set_seed(ckpt_seed + mode_offset)
                    acc = evaluate_checkpoint(
                        model, grammar, eval_ds, cfg, schedule, device, temperature
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
            epochs_str  = ','.join(str(int(p.name.split('=')[1])) for p in checkpoints)

            all_csv_rows.append({
                'dataset':                      ds_type,
                'sweep_label':                  label,
                'schedule':                     getattr(getattr(cfg, 'schedule', None), 'type', 'categorical'),
                'sigma':                        getattr(getattr(cfg, 'schedule', None), 'sigma', ''),
                'T':                            cfg.model.T,
                'n_checkpoints_used':           len(checkpoints),
                'epochs_evaluated':             epochs_str,
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
    print('RESULTS')
    print('=' * 60)

    table_lines = []
    for ds_type in DATASET_TYPES:
        rows = per_dataset_table_rows[ds_type]
        actual_n = eval_datasets[ds_type].data.shape[0]
        table = make_table(rows, ds_type, actual_n)
        print(table)
        table_lines.append(table)

    if all_csv_rows:
        csv_path = save_csv(all_csv_rows, out_dir)

    table_path = out_dir / 'sweep_eval_table.txt'
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines) + '\n')
    print(f"Table saved to: {table_path}")


if __name__ == '__main__':
    main()
