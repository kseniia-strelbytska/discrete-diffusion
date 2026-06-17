"""
benchmark_unmasking.py
======================
Tracks per-timestep uncertainty fraction during Gaussian-schedule denoising
using the real grammar oracle (oracleModel from oracle/grammar_oracles.py).

The oracle is a drop-in replacement for a trained model: given a partially-
unmasked sequence it returns the exact marginal probability distribution over
each position.  No model training required.

Certain token:   max content prob >= CONFIDENCE_THRESHOLD
Uncertain token: max content prob <  CONFIDENCE_THRESHOLD

If zero tokens are unmasked at a given step, that step's fraction is 0.0.

Usage
-----
  python scripts/benchmark_unmasking.py                      # default: anbn
  python scripts/benchmark_unmasking.py --grammar baN
  python scripts/benchmark_unmasking.py --grammar aNbNcN --grammar-l 128

Available grammars: anbn, baN, bbaN, aNbNcN,
                    not_nested_parentheses_and_brackets,
                    parentheses_and_brackets

Outputs  (results/unmasked_tokens_per_step/<grammar>/):
  uncertainty_T{T}.csv / .txt   — Sigma vs overall uncertainty fraction
  uncertainty_T{T}.png          — 2×3 grid of per-timestep uncertainty plots
"""

import sys
import os
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import csv

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from schedules.gaussian_schedule import GaussianSchedule
from oracle.grammar_oracles import oracleModel, _VOCAB_SIZE_MAP
from datasets.constants import SOS_token, MASK_token

# ── Fixed configuration ────────────────────────────────────────────────────────

T_VALUES             = [64, 128, 512]
SIGMA_VALUES         = [1, 4, 20, 40, 60, 100, 140, 190, 240]
SEEDS                = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # for averaging over multiple runs
CONFIDENCE_THRESHOLD = 0.8

# Grammar names accepted on the CLI
GRAMMAR_CHOICES = [
    "anbn",
    "baN",
    "bbaN",
    "aNbNcN",
    "not_nested_parentheses_and_brackets",
    "parentheses_and_brackets",
]

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark unmasking uncertainty via grammar oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grammar", default="anbn", choices=GRAMMAR_CHOICES,
        help="Grammar whose oracle is used for token probabilities.",
    )
    p.add_argument(
        "--grammar-l", type=int, default=256,
        help="Sequence length L (number of positions incl. SOS/EOS/PAD).",
    )
    return p.parse_args()

# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_trajectory(
    T: int,
    sigma: float,
    seed: int,
    grammar_name: str,
    vocab_size: int,
    L: int,
    device: torch.device,
):
    """
    Full denoising trajectory with per-step uncertainty tracking.

    Mirrors ScheduledUnmasker.forward (denoise="0", oracle=True):
      - linspace timesteps 1.0 → 0.0, dt = 1/T
      - alpha_t / alpha_s from GaussianSchedule.p_mask
      - weight = (α_s − α_t) / (1 − α_t), mask_prob = (1 − α_s) / (1 − α_t)
      - oracle marginals used directly as content_probs (no softmax — they are
        already probabilities, matching the oracle=True branch in ScheduledUnmasker)
      - full probability vector built and sampled via multinomial

    Returns
    -------
    per_step_frac : list[float]  length T; uncertain/total at each step (0.0 if none)
    n_uncertain   : int          total uncertain tokens over the trajectory
    n_unmasked    : int          total unmasked tokens over the trajectory
    """
    torch.manual_seed(seed)

    schedule  = GaussianSchedule(sigma)
    oracle    = oracleModel(grammar_name=grammar_name, vocab_size=vocab_size, device=device)
    eps       = 1e-5
    timesteps = torch.linspace(1.0, 0.0, T + 1, device=device)

    # content_t: all token indices except MASK (oracle returns full vocab marginals)
    content_t = torch.tensor(
        [i for i in range(vocab_size) if i != MASK_token],
        dtype=torch.long, device=device,
    )

    # Starting sequence: SOS at position 0, everything else MASK
    X      = torch.full((L,), MASK_token, dtype=torch.long, device=device)
    X[0]   = SOS_token
    masked = (X == MASK_token)   # position 0 stays False throughout

    per_step_frac = []
    n_uncertain   = 0
    n_unmasked    = 0

    for i in range(T):
        t = timesteps[i]
        s = timesteps[i + 1]

        if t <= 0:
            per_step_frac.append(0.0)
            continue

        # ── Noise schedule ────────────────────────────────────────────────────
        p_mask_t = schedule.p_mask(t, max_l=L, device=device)           # (1, L)
        p_mask_s = schedule.p_mask(s, max_l=L, device=device)           # (1, L)
        alpha_t  = (1.0 - p_mask_t).squeeze(0)                          # (L,)
        alpha_s  = (1.0 - p_mask_s).squeeze(0)

        denom     = (1.0 - alpha_t).clamp(min=eps)
        weight    = ((alpha_s - alpha_t) / denom).clamp(min=0.0)        # (L,)
        mask_prob = ((1.0 - alpha_s)     / denom).clamp(min=0.0, max=1.0)

        # ── Oracle forward ────────────────────────────────────────────────────
        # oracle returns marginal probabilities (L, vocab_size); no softmax needed
        # (matches the oracle=True branch in ScheduledUnmasker).
        # grammar_oracles.py accumulates counts as Python ints and divides before
        # converting to a tensor, so there is no overflow even for 2^k completions.
        try:
            marginals     = oracle(X).float()                           # (L, V)
            content_probs = marginals[:, content_t]                     # (L, V-1)
        except ValueError:
            # Sequence became structurally invalid — skip step.
            per_step_frac.append(0.0)
            continue

        # ── Full probability distribution (mirrors ScheduledUnmasker) ─────────
        probs = torch.zeros(L, vocab_size, device=device)
        probs[:, content_t]  = content_probs * weight.unsqueeze(-1)
        probs[:, MASK_token] = mask_prob

        probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        zero_rows = probs.sum(dim=-1) == 0
        probs[zero_rows, MASK_token] = 1.0

        # ── Sample and record uncertainty ─────────────────────────────────────
        sampled        = torch.multinomial(probs, 1).squeeze(-1)        # (L,)
        newly_unmasked = masked & (sampled != MASK_token)
        count          = int(newly_unmasked.sum().item())

        if count == 0:
            per_step_frac.append(0.0)
        else:
            max_probs = content_probs[newly_unmasked].max(dim=-1).values  # (count,)
            uncertain = int((max_probs < CONFIDENCE_THRESHOLD).sum().item())
            per_step_frac.append(uncertain / count)
            n_uncertain += uncertain
            n_unmasked  += count

        # Update sequence: newly unmasked positions take their sampled token
        X[newly_unmasked] = sampled[newly_unmasked]
        masked = (X == MASK_token)

    return per_step_frac, n_uncertain, n_unmasked


def run_all(grammar_name: str, vocab_size: int, L: int, device: torch.device) -> dict:
    """
    Returns results[(T, sigma)] = {"per_step": np.ndarray(T,), "overall": float}
    """
    results = {}
    total   = len(T_VALUES) * len(SIGMA_VALUES)
    done    = 0

    for T in T_VALUES:
        for sigma in SIGMA_VALUES:
            seed_fracs = []
            agg_unc    = 0
            agg_tot    = 0

            for seed in SEEDS:
                frac, unc, tot = simulate_trajectory(
                    T, sigma, seed, grammar_name, vocab_size, L, device
                )
                seed_fracs.append(frac)
                agg_unc += unc
                agg_tot += tot

            avg_per_step = np.mean(seed_fracs, axis=0)
            overall      = agg_unc / agg_tot if agg_tot > 0 else 0.0

            results[(T, sigma)] = {"per_step": avg_per_step, "overall": overall}
            done += 1
            print(f"  [{done:2d}/{total}]  T={T:4d}  sigma={sigma:5.1f}"
                  f"  overall_uncertainty={overall:.4f}")

    return results

# ── Tables ─────────────────────────────────────────────────────────────────────

def save_tables(results: dict, T: int, out_dir: str) -> None:
    rows = [(sigma, results[(T, sigma)]["overall"]) for sigma in SIGMA_VALUES]
    base = os.path.join(out_dir, f"uncertainty_T{T}")

    with open(base + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "overall_avg_uncertainty_fraction"])
        writer.writerows(rows)
    print(f"  Saved {base}.csv")

    header = f"  T = {T}\n  {'Sigma':>8}  {'Overall Avg Uncertainty Fraction':>34}"
    sep    = "  " + "-" * 45
    lines  = [header, sep]
    for sigma, frac in rows:
        lines.append(f"  {sigma:>8.1f}  {frac:>34.6f}")
    with open(base + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {base}.txt")

# ── Plots ──────────────────────────────────────────────────────────────────────

def save_plot(results: dict, T: int, grammar_name: str, L: int, out_dir: str) -> None:
    n_sigmas = len(SIGMA_VALUES)
    if n_sigmas == 0:
        return

    # Define layout grid properties
    ncols = 3
    nrows = int(np.ceil(n_sigmas / ncols))
    
    # squeeze=False ensures `axes` is always a 2D numpy array, 
    # even if nrows=1 or ncols=1. This prevents errors when calling .flatten()
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(
        f"Per-Timestep Uncertainty Fraction  |  grammar={grammar_name}  T={T}  L={L}\n"
        f"Certain ≡ max content prob ≥ {CONFIDENCE_THRESHOLD}   "
        f"({len(SEEDS)} seeds, oracle model)",
        fontsize=12,
    )

    x_steps = np.arange(T, 0, -1)   # [T, T-1, ..., 1], length T

    for i, sigma in enumerate(SIGMA_VALUES):
        ax = axes_flat[i]
        per_step = results[(T, sigma)]["per_step"]   # (T,)
        overall  = results[(T, sigma)]["overall"]

        ax.plot(x_steps, per_step,
                linewidth=0.9, color="steelblue", alpha=0.85, label="per-step frac")
        ax.axhline(overall, color="tomato", linestyle="--", linewidth=1.3,
                   label=f"traj avg = {overall:.3f}")

        ax.set_xlim(T, 0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"σ = {sigma}", fontsize=11)
        ax.set_xlabel("Timestep  (T → 0)", fontsize=9)
        ax.set_ylabel("Uncertainty Fraction", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)

    # Clean up: Hide any unused empty subplots in the grid
    for j in range(n_sigmas, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"uncertainty_T{T}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    grammar_name = args.grammar
    L            = args.grammar_l

    # Grammar-specific vocab size (floored to the grammar's minimum)
    resolved     = "aNbN" if grammar_name == "anbn" else grammar_name
    vocab_size   = _VOCAB_SIZE_MAP.get(resolved, 6)

    out_dir = os.path.join(REPO_ROOT, "results", "unmasked_tokens_per_step", grammar_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device      : {device}")
    print(f"Grammar     : {grammar_name}  (vocab_size={vocab_size})")
    print(f"L={L}  T={T_VALUES}  sigma={SIGMA_VALUES}")
    print(f"Seeds={SEEDS}  confidence_threshold={CONFIDENCE_THRESHOLD}\n")

    results = run_all(grammar_name, vocab_size, L, device)

    print("\nSaving outputs...")
    for T in T_VALUES:
        save_tables(results, T, out_dir)
        save_plot(results, T, grammar_name, L, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
