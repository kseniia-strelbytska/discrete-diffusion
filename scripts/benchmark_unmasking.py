"""
Benchmark: average number of tokens unmasked per denoising step.

No model forward passes — unmasking counts come purely from the Gaussian
noise schedule probabilities.  The stochastic mask/unmask decision at each
step is replicated over NUM_SEEDS independent seeds so the reported mean is
reliable.

Outputs (results/unmasked_tokens_per_step/):
  - data.csv           — raw table of (T, sigma, avg_unmasked_per_step)
  - data.txt           — same table, pretty-printed
  - plot.png           — 3-row subplot, one per T value
"""

import sys
import os

# Allow imports from src/ regardless of CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import csv
import math
import textwrap

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text  # pip install adjustText — soft dep, see fallback below

from schedules.gaussian_schedule import GaussianSchedule

# ── Configuration ────────────────────────────────────────────────────────────

L          = 256
T_VALUES   = [128, 256, 512]
SIGMA_VALUES = [0.5, 1, 2, 4, 10, 20, 40, 60, 80, 100, 120, 140, 160, 200, 240]
NUM_SEEDS  = 100

MASK_TOKEN = 5   # matches datasets.constants.MASK_token

OUT_DIR    = os.path.join(REPO_ROOT, "results", "unmasked_tokens_per_step")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Core simulation ──────────────────────────────────────────────────────────

def simulate_denoising(T: int, sigma: float, seed: int, device: torch.device) -> float:
    """
    Run one full denoising trajectory (T → 0) for a fully-masked sequence of
    length L and return the average number of tokens unmasked per step.

    The simulation mirrors the 'denoise="0"' branch in ScheduledUnmasker.forward:
      - timesteps are evenly spaced from 1.0 down to 0.0 with dt = 1/T
      - at each step the retention probabilities α_t and α_s are computed from
        the GaussianSchedule
      - a token that is still masked is unmasked with probability
            (α_s - α_t) / (1 - α_t)
        and remains masked with probability
            (1 - α_s) / (1 - α_t)
      - no model logits are needed; we only track whether a MASK token
        transitions to a non-MASK token
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    schedule = GaussianSchedule(sigma)

    eps = 1e-5
    dt  = 1.0 / T

    # Start fully masked.
    masked = torch.ones(L, dtype=torch.bool, device=device)   # True ↔ still masked

    total_unmasked = 0

    timesteps = torch.linspace(1.0, 0.0, T + 1, device=device)

    for i in range(T):
        t = timesteps[i]
        s = timesteps[i + 1]   # s = t - dt

        if t <= 0:
            break

        # p_mask returns shape (1, L); squeeze to (L,)
        p_mask_t = schedule.p_mask(t, max_l=L, device=device).squeeze(0)
        p_mask_s = schedule.p_mask(s, max_l=L, device=device).squeeze(0)

        alpha_t = 1.0 - p_mask_t   # shape (L,)
        alpha_s = 1.0 - p_mask_s

        # Probability that a currently-masked token gets unmasked this step.
        # Formula: (α_s - α_t) / (1 - α_t), clamped to [0, 1].
        denom      = (1.0 - alpha_t).clamp(min=eps)
        p_unmask   = ((alpha_s - alpha_t) / denom).clamp(min=0.0, max=1.0)

        # Sample: for each still-masked position, does it unmask?
        rand_vals  = torch.rand(L, device=device)
        newly_unmasked = masked & (rand_vals < p_unmask)

        count = newly_unmasked.sum().item()
        total_unmasked += count
        masked[newly_unmasked] = False

    avg_per_step = total_unmasked / T
    return avg_per_step


def run_all_configs(device: torch.device) -> list[dict]:
    """
    Iterate over every (T, sigma) pair, average over NUM_SEEDS seeds, and
    return a list of result dicts.
    """
    seeds = list(range(NUM_SEEDS))
    results = []

    total = len(T_VALUES) * len(SIGMA_VALUES)
    done  = 0
    for T in T_VALUES:
        for sigma in SIGMA_VALUES:
            seed_avgs = []
            for seed in seeds:
                avg = simulate_denoising(T, sigma, seed, device)
                seed_avgs.append(avg)
            mean_avg = float(np.mean(seed_avgs))
            results.append({"T": T, "sigma": sigma, "avg_unmasked_per_step": mean_avg})
            done += 1
            print(f"  [{done:3d}/{total}]  T={T:4d}  sigma={sigma:6.1f}  avg={mean_avg:.4f}")

    return results


# ── I/O helpers ─────────────────────────────────────────────────────────────

def save_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "sigma", "avg_unmasked_per_step"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV  → {path}")


def save_txt(results: list[dict], path: str) -> None:
    header = f"{'T':>6}  {'sigma':>8}  {'avg_unmasked_per_step':>22}"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for r in results:
        lines.append(f"{r['T']:>6}  {r['sigma']:>8.1f}  {r['avg_unmasked_per_step']:>22.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved TXT  → {path}")


# ── Plotting ─────────────────────────────────────────────────────────────────

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">", "H", "8", "d"]

def make_plot(results: list[dict], path: str) -> None:
    # Reorganise into {T: [(sigma, avg), ...]}
    data_by_T: dict[int, list] = {T: [] for T in T_VALUES}
    for r in results:
        data_by_T[r["T"]].append((r["sigma"], r["avg_unmasked_per_step"]))
    for T in T_VALUES:
        data_by_T[T].sort(key=lambda x: x[0])

    fig, axes = plt.subplots(len(T_VALUES), 1, figsize=(10, 5 * len(T_VALUES)))
    if len(T_VALUES) == 1:
        axes = [axes]

    fig.suptitle(
        f"Gaussian Schedule — Average Tokens Unmasked Per Step\n"
        f"(L={L}, {NUM_SEEDS} seeds per config)",
        fontsize=14, y=1.01
    )

    for ax, T in zip(axes, T_VALUES):
        pairs   = data_by_T[T]
        sigmas  = [p[0] for p in pairs]
        avgs    = [p[1] for p in pairs]

        # Line + markers
        ax.plot(sigmas, avgs, linestyle="-", linewidth=1.2, color="steelblue", zorder=2)
        for idx, (sig, avg) in enumerate(pairs):
            ax.scatter(sig, avg, marker=MARKERS[idx % len(MARKERS)],
                       s=70, zorder=3, color="steelblue", edgecolors="white", linewidths=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Sigma (log scale)", fontsize=11)
        ax.set_ylabel("Avg Unmasked Tokens / Step", fontsize=11)
        ax.set_title(f"T = {T}", fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(sigmas)
        ax.tick_params(axis="x", rotation=30)

        # Annotate each point with its (sigma, avg) value.
        texts = []
        for sig, avg in pairs:
            label = f"({sig:.4g}, {avg:.2f})"
            txt = ax.text(sig, avg, label, fontsize=7.5, ha="center", va="bottom")
            texts.append(txt)

        # Try to use adjustText to reduce overlaps; fall back gracefully.
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
        except Exception:
            pass

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"L={L}, T values={T_VALUES}, sigma values={SIGMA_VALUES}, seeds={NUM_SEEDS}\n")

    results = run_all_configs(device)

    csv_path  = os.path.join(OUT_DIR, "data.csv")
    txt_path  = os.path.join(OUT_DIR, "data.txt")
    plot_path = os.path.join(OUT_DIR, "plot.png")

    save_csv(results, csv_path)
    save_txt(results, txt_path)
    make_plot(results, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
