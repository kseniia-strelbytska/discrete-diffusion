#!/usr/bin/env python3
"""
analyze_t_sweep.py
==================
Research-paper figures for the discrete-diffusion T-sweep over formal grammars.

Answers three questions, one deliverable each:

  Q1  How does accuracy change with mean number of steps made (compute)?
      Does a higher T always mean better accuracy?
        -> fig1_accuracy_vs_compute.png
           6 panels (one per grammar), 8 lines (decoder x sampler). The best
           hyperparameter is chosen INDEPENDENTLY at each compute budget, so a line is
           the best-achievable-accuracy envelope and adjacent points may use different
           sigma/gamma. x = mean steps (compute, log), y = accuracy.

  Q2  How does diversity change with accuracy?
        -> fig2_diversity_vs_accuracy.png
           Same 6 x 8 layout and the SAME per-budget hyperparameter selection as Q1.
           x = accuracy, y = grammar-specific diversity metric.
           Each line traces the sweep trajectory (ordered by compute).

  Q3  Greedy vs categorical -- is one always better for diversity vs accuracy?
        -> fig3_paired_deltas.png        (headline: paired Delta-acc / Delta-div)
           fig3b_tradeoff_quadrant.png   (companion: Delta-acc vs Delta-div scatter)
        Greedy and categorical are evaluated at identical cells, so they are
        compared pairwise. A zero reference line makes "always better?" readable.

Usage
-----
    python analyze_t_sweep.py --file results.csv
    python analyze_t_sweep.py --file results.csv --compute-mode mixed --xscale log

The script processes the CSV with pandas and plots with seaborn.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
# Configuration -- edit here if your column names differ.                     #
# --------------------------------------------------------------------------- #

# Logical role -> candidate column names (first match wins, case-insensitive).
COL_CANDIDATES = {
    "grammar":   ["grammar", "dataset", "language", "task"],
    "decoder":   ["strategy", "decoder", "schedule", "noise_schedule"],
    "sampler":   ["sampling_strategy", "sampler", "sampling"],
    "T":         ["T", "model_T", "steps_requested", "num_steps"],
    "compute":   ["n_steps_mean", "mean_steps", "nfe", "n_steps", "mean_nfe", "avg_steps"],
    "accuracy":  ["mean_both_rules", "both_rules_acc", "accuracy", "acc", "mean_acc"],
    "sigma":     ["sigma"],
    "eb_gamma":  ["eb_gamma", "gamma"],
    "too_low":   ["n_correct_too_low", "too_few_correct", "diversity_unreliable"],
}

# decoder value -> the hyperparameter column it sweeps (others: no hyperparameter)
HPARAM_COL = {"gaussian": "sigma", "eb": "eb_gamma"}

# Per-grammar diversity metric(s). First entry is the primary axis for Q2.
DIVERSITY_METRICS = {
    "baN": ["uniqueness"],
    "bbaN": ["nm_joint_coverage"],
    "aNbN": ["n_coverage"],
    "aNbNcN": ["n_entropy", "n_coverage"],
    "parentheses_and_brackets": ["uniqueness"],
    "not_nested_parentheses_and_brackets": ["uniqueness"],
}

# Canonical ordering / display names so colours stay stable across figures.
DECODER_ORDER = ["uniform", "gaussian", "eb", "ar"]
DECODER_DISPLAY = {"uniform": "Uniform", "gaussian": "Gaussian", "eb": "EB", "ar": "AR"}
SAMPLER_ORDER = ["categorical", "greedy"]
SAMPLER_DISPLAY = {"categorical": "Categorical", "greedy": "Greedy"}
# greedy = solid, categorical = dashed; distinct markers too.
SAMPLER_DASHES = {"greedy": "", "categorical": (4, 1.5)}
SAMPLER_MARKERS = {"greedy": "o", "categorical": "X"}

GRAMMAR_DISPLAY = {
    "baN": "baᴺ",
    "bbaN": "bbaᴺ",
    "aNbN": "aᴺbᴺ",
    "aNbNcN": "aᴺbᴺcᴺ",
    "parentheses_and_brackets": "Dyck-2 (nested)",
    "not_nested_parentheses_and_brackets": "Dyck-2 (flat)",
}


# --------------------------------------------------------------------------- #
# Column resolution                                                           #
# --------------------------------------------------------------------------- #

def resolve_columns(df):
    """Map each logical role to a real column; error clearly if a required one is absent."""
    lower = {c.lower(): c for c in df.columns}
    resolved = {}
    for role, candidates in COL_CANDIDATES.items():
        for cand in candidates:
            if cand.lower() in lower:
                resolved[role] = lower[cand.lower()]
                break
    required = ["grammar", "decoder", "sampler", "compute", "accuracy"]
    missing = [r for r in required if r not in resolved]
    if missing:
        sys.exit(
            "ERROR: could not find required column(s) for role(s): "
            f"{missing}\nAvailable columns:\n  " + "\n  ".join(df.columns)
            + "\nEdit COL_CANDIDATES at the top of the script to match your CSV."
        )
    return resolved


# --------------------------------------------------------------------------- #
# Load & prepare                                                              #
# --------------------------------------------------------------------------- #

def load_and_prepare(path, diversity_index):
    df = pd.read_csv(path)
    R = resolve_columns(df)

    # Normalise the working frame to canonical column names.
    work = pd.DataFrame()
    work["grammar"] = df[R["grammar"]].astype(str)
    work["decoder"] = df[R["decoder"]].astype(str).str.lower().str.strip()
    work["sampler"] = df[R["sampler"]].astype(str).str.lower().str.strip()
    work["compute"] = pd.to_numeric(df[R["compute"]], errors="coerce")
    work["accuracy"] = pd.to_numeric(df[R["accuracy"]], errors="coerce")
    work["T"] = pd.to_numeric(df[R["T"]], errors="coerce") if "T" in R else np.nan

    for h in ("sigma", "eb_gamma"):
        work[h] = pd.to_numeric(df[R[h]], errors="coerce") if h in R else np.nan

    # Single hyperparameter column keyed by decoder (NaN where the decoder has none).
    work["hparam"] = np.where(
        work["decoder"] == "gaussian", work["sigma"],
        np.where(work["decoder"] == "eb", work["eb_gamma"], np.nan),
    )

    # Per-grammar diversity value on a single axis.
    def metric_for(grammar):
        ms = DIVERSITY_METRICS.get(grammar, [])
        if not ms:
            return None
        return ms[min(diversity_index, len(ms) - 1)]

    work["diversity_metric"] = work["grammar"].map(metric_for)

    def gather(row):
        m = row["diversity_metric"]
        if m is None or m not in df.columns:
            return np.nan
        return pd.to_numeric(pd.Series([df.loc[row.name, m]]), errors="coerce").iloc[0]

    work["diversity"] = work.apply(gather, axis=1)

    # Respect the "too few correct -> diversity unreliable" flag.
    if "too_low" in R:
        flag = df[R["too_low"]].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
        work.loc[flag.values, "diversity"] = np.nan

    return work


# --------------------------------------------------------------------------- #
# Best-hyperparameter selection, PER COMPUTE BUDGET (for Q1 / Q2 lines)        #
# --------------------------------------------------------------------------- #

def add_compute_axis(df, compute_mode):
    """Attach `x_compute`, the value used on the compute axis (Q1) and as the budget.

    compute_mode:
      'steps'  -> realised mean steps (NFE) for every decoder. Apples-to-apples compute,
                  and it matches the Q1 wording ("mean number of steps made").  [recommended]
      'mixed'  -> realised mean steps for EB (it is fully adaptive, so its T is meaningless)
                  but the requested T for every other decoder.
      'T'      -> requested T for every decoder.
    """
    df = df.copy()
    if compute_mode == "steps":
        df["x_compute"] = df["compute"]
    elif compute_mode == "T":
        df["x_compute"] = df["T"]
    else:  # mixed
        df["x_compute"] = np.where(df["decoder"] == "eb", df["compute"], df["T"])
    return df


def select_best_per_budget(df):
    """Pick the best hyperparameter *independently at each compute budget*.

    For one (grammar, decoder, sampler), every (hyperparameter, T) row is a candidate.
    Candidates are bucketed by compute budget (rounded `x_compute`) and, within each
    bucket, the single setting with the highest accuracy is kept. A line is therefore the
    best-achievable-accuracy envelope over hyperparameters as a function of compute --
    adjacent points may come from different sigma / gamma.

    For EB (x_compute = realised steps) gamma both sets the compute and is the thing being
    optimised, so this traces EB's accuracy-vs-steps frontier; redundant T rows collapse
    into the same bucket. For fixed schedules (x_compute = T) sigma is optimised at each T.
    Hyperparameter-free decoders have one candidate per bucket and pass through unchanged.
    """
    d = df.dropna(subset=["accuracy", "x_compute"]).copy()
    d["_budget"] = d["x_compute"].round()
    keys = ["grammar", "decoder", "sampler", "_budget"]
    idx = d.groupby(keys, sort=False)["accuracy"].idxmax()
    best = d.loc[idx].drop(columns="_budget").reset_index(drop=True)
    return best


# --------------------------------------------------------------------------- #
# Shared plotting helpers                                                      #
# --------------------------------------------------------------------------- #

def present_levels(series, order):
    present = [v for v in order if v in set(series)]
    # append any unexpected values so nothing is silently dropped
    present += [v for v in series.unique() if v not in present]
    return present


def style_kwargs(df):
    decoders = present_levels(df["decoder"], DECODER_ORDER)
    samplers = present_levels(df["sampler"], SAMPLER_ORDER)
    palette = dict(zip(decoders, sns.color_palette("colorblind", len(decoders))))
    dashes = {s: SAMPLER_DASHES.get(s, "") for s in samplers}
    markers = {s: SAMPLER_MARKERS.get(s, "o") for s in samplers}
    return decoders, samplers, palette, dashes, markers


def grammar_order(df):
    return present_levels(df["grammar"], list(DIVERSITY_METRICS.keys()))


def relabel_legend(g):
    """Rewrite the relplot legend with human-readable decoder / sampler names."""
    if g.legend is None:
        return
    for txt in g.legend.texts:
        s = txt.get_text()
        if s in DECODER_DISPLAY:
            txt.set_text(DECODER_DISPLAY[s])
        elif s in SAMPLER_DISPLAY:
            txt.set_text(SAMPLER_DISPLAY[s])
        elif s == "decoder":
            txt.set_text("Decoder")
        elif s == "sampler":
            txt.set_text("Sampler")


# --------------------------------------------------------------------------- #
# Q1: accuracy vs compute                                                     #
# --------------------------------------------------------------------------- #

def fig_accuracy_vs_compute(df, outpath, dpi, xscale, compute_mode):
    d = df.dropna(subset=["x_compute", "accuracy"]).copy()
    d = d[d["x_compute"] > 0]
    d = d.sort_values(["grammar", "decoder", "sampler", "x_compute"])

    gorder = grammar_order(d)
    decoders, samplers, palette, dashes, markers = style_kwargs(d)

    g = sns.relplot(
        data=d, x="x_compute", y="accuracy",
        hue="decoder", style="sampler",
        hue_order=decoders, style_order=samplers,
        palette=palette, dashes=dashes, markers=markers,
        col="grammar", col_order=gorder, col_wrap=3,
        kind="line", estimator=None, sort=True,
        height=3.0, aspect=1.25,
        facet_kws=dict(sharex=True, sharey=True),   # uniform scale across panels
        markersize=6, linewidth=1.6,
    )
    for grammar, ax in g.axes_dict.items():
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(GRAMMAR_DISPLAY.get(grammar, grammar), fontsize=11)
        ax.grid(True, which="major", ls="--", alpha=0.35)
    xlabel = {
        "steps": "Mean steps made  (compute = NFE, all decoders)",
        "mixed": "Compute  —  realised mean steps for EB, requested T otherwise",
        "T": "Requested steps T  (all decoders)",
    }[compute_mode]
    if xscale == "log":
        xlabel += "  (log)"
    g.set_axis_labels(xlabel, "Both-rules accuracy")
    relabel_legend(g)
    g.figure.suptitle("Accuracy vs. compute  (best hyperparameter chosen at each compute budget)",
                      fontsize=13, y=1.02)
    g.figure.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)


# --------------------------------------------------------------------------- #
# Q2: diversity vs accuracy                                                   #
# --------------------------------------------------------------------------- #

def fig_diversity_vs_accuracy(df, outpath, dpi, normalize):
    d = df.dropna(subset=["accuracy", "diversity"]).copy()
    # Order points along each line by the swept compute, so non-monotonic trajectories
    # (e.g. categorical's U-shape) are drawn faithfully rather than re-sorted by accuracy.
    d = d.sort_values(["grammar", "decoder", "sampler", "x_compute"])

    metric_by_grammar = d.groupby("grammar")["diversity_metric"].first().to_dict()

    if normalize:
        # Min-max normalise within each grammar so all panels share one [0,1] scale.
        d["y_div"] = d.groupby("grammar")["diversity"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
        )
        ylabel = "Diversity  (min-max normalised within grammar)"
    else:
        d["y_div"] = d["diversity"]
        ylabel = "Diversity (per-grammar metric)"

    gorder = grammar_order(d)
    decoders, samplers, palette, dashes, markers = style_kwargs(d)

    g = sns.relplot(
        data=d, x="accuracy", y="y_div",
        hue="decoder", style="sampler",
        hue_order=decoders, style_order=samplers,
        palette=palette, dashes=dashes, markers=markers,
        col="grammar", col_order=gorder, col_wrap=3,
        kind="line", estimator=None, sort=False,
        height=3.0, aspect=1.25,
        # uniform scale: share both axes when normalised; otherwise free y per metric.
        facet_kws=dict(sharex=True, sharey=normalize),
        markersize=6, linewidth=1.6,
    )
    for grammar, ax in g.axes_dict.items():
        metric = metric_by_grammar.get(grammar, "diversity")
        ax.set_title(f"{GRAMMAR_DISPLAY.get(grammar, grammar)}", fontsize=11)
        ax.set_xlabel("Both-rules accuracy")
        ax.set_xlim(-0.03, 1.03)
        if normalize:
            ax.set_ylim(-0.03, 1.03)
            # keep the actual metric visible as a small in-panel note
            ax.text(0.03, 0.97, metric, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color="dimgrey")
        else:
            ax.set_ylabel(metric)
        ax.grid(True, which="major", ls="--", alpha=0.35)
    if normalize:
        g.set_ylabels(ylabel)
    relabel_legend(g)
    g.figure.suptitle("Diversity vs. accuracy  (same per-budget selection as Q1; line traces compute)",
                      fontsize=13, y=1.02)
    g.figure.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)


# --------------------------------------------------------------------------- #
# Q3: greedy vs categorical -- paired deltas                                  #
# --------------------------------------------------------------------------- #

def build_pairs(df):
    """Match greedy vs categorical at identical (grammar, decoder, hparam, T) cells."""
    df = df.copy()
    df["cell"] = (
        df["grammar"].astype(str) + "|" + df["decoder"].astype(str) + "|"
        + df["hparam"].round(6).astype(str) + "|" + df["T"].astype(str)
    )
    keep = ["grammar", "decoder", "hparam", "T", "cell", "accuracy", "diversity"]
    gr = df[df["sampler"] == "greedy"][keep]
    ca = df[df["sampler"] == "categorical"][keep]
    m = gr.merge(ca, on=["grammar", "decoder", "hparam", "T", "cell"],
                 suffixes=("_greedy", "_categorical"))
    m["d_acc"] = m["accuracy_greedy"] - m["accuracy_categorical"]
    m["d_div"] = m["diversity_greedy"] - m["diversity_categorical"]
    return m


def fig_paired_deltas(pairs, outpath, dpi):
    gorder = grammar_order(pairs)
    pairs = pairs.copy()
    pairs["grammar_disp"] = pairs["grammar"].map(lambda x: GRAMMAR_DISPLAY.get(x, x))
    disp_order = [GRAMMAR_DISPLAY.get(g, g) for g in gorder]
    decoders = present_levels(pairs["decoder"], DECODER_ORDER)
    palette = dict(zip(decoders, sns.color_palette("colorblind", len(decoders))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, col, title, sign in [
        (axes[0], "d_acc", "Δ accuracy  (greedy − categorical)", "above 0  ⇒  greedy more accurate"),
        (axes[1], "d_div", "Δ diversity  (greedy − categorical)", "below 0  ⇒  categorical more diverse"),
    ]:
        sub = pairs.dropna(subset=[col])
        sns.stripplot(
            data=sub, x="grammar_disp", y=col, order=disp_order,
            hue="decoder", hue_order=decoders, palette=palette,
            dodge=True, jitter=0.18, size=4, alpha=0.7, ax=ax, legend=(ax is axes[0]),
        )
        # per-grammar mean marker
        means = sub.groupby("grammar_disp")[col].mean().reindex(disp_order)
        ax.scatter(range(len(disp_order)), means.values, marker="_", s=600,
                   color="black", linewidth=2.2, zorder=5)
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"{title}\n{sign}", fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel(col.replace("d_", "Δ "))
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    if axes[0].get_legend() is not None:
        for t in axes[0].get_legend().texts:
            t.set_text(DECODER_DISPLAY.get(t.get_text(), t.get_text()))
        axes[0].legend(title="Decoder", fontsize=8, title_fontsize=9, loc="best")

    fig.suptitle("Greedy vs. categorical at matched cells  (black bar = per-grammar mean)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff_quadrant(pairs, outpath, dpi):
    sub = pairs.dropna(subset=["d_acc", "d_div"]).copy()
    if sub.empty:
        return False
    gorder = grammar_order(sub)
    sub["grammar_disp"] = sub["grammar"].map(lambda x: GRAMMAR_DISPLAY.get(x, x))
    disp_order = [GRAMMAR_DISPLAY.get(g, g) for g in gorder]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.scatterplot(
        data=sub, x="d_acc", y="d_div", hue="grammar_disp", hue_order=disp_order,
        style="grammar_disp", style_order=disp_order, s=70, ax=ax,
        palette="colorblind",
    )
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", lw=1)
    xmax = np.nanmax(np.abs(sub["d_acc"])) * 1.15 or 1
    ymax = np.nanmax(np.abs(sub["d_div"])) * 1.15 or 1
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.text(xmax * 0.96, -ymax * 0.96, "greedy: +acc / −div",
            ha="right", va="bottom", fontsize=8, color="dimgrey")
    ax.set_xlabel("Δ accuracy  (greedy − categorical)")
    ax.set_ylabel("Δ diversity  (greedy − categorical)")
    ax.set_title("Joint accuracy–diversity trade-off per matched cell")
    ax.legend(title="Grammar", fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Console summary that helps with the write-up                                #
# --------------------------------------------------------------------------- #

def print_summary(best_df, pairs):
    print("\n" + "=" * 70)
    print("AUTOMATED SUMMARY  (sanity-check for the prose; verify before quoting)")
    print("=" * 70)

    # Q1: is more compute always better?
    print("\n[Q1] Does higher compute always improve accuracy?")
    nonmono = []
    for (g, d, s), grp in best_df.dropna(subset=["x_compute", "accuracy"]).groupby(
        ["grammar", "decoder", "sampler"], sort=False
    ):
        grp = grp.sort_values("x_compute")
        acc = grp["accuracy"].values
        if len(acc) < 2:
            continue
        peak_at_end = np.argmax(acc) == len(acc) - 1
        if not peak_at_end and (acc.max() - acc[-1] > 1e-9):
            nonmono.append((g, d, s, float(acc[-1]), float(acc.max())))
    if nonmono:
        print(f"  No -- {len(nonmono)} decoder x sampler line(s) peak at intermediate compute:")
        for g, d, s, last, peak in nonmono[:12]:
            print(f"    {g:>32s} | {d:<8s} {s:<11s} acc@max-compute={last:.3f} vs peak={peak:.3f}")
    else:
        print("  Within these lines accuracy is non-decreasing in compute.")

    if pairs.empty:
        print("\n[Q3] No matched greedy/categorical cells found.")
        return

    # Q3: paired verdicts
    print("\n[Q3] Greedy vs categorical at matched cells:")
    acc_pairs = pairs.dropna(subset=["d_acc"])
    div_pairs = pairs.dropna(subset=["d_div"])
    if len(acc_pairs):
        frac_g = (acc_pairs["d_acc"] > 0).mean()
        print(f"  Accuracy: greedy > categorical in {frac_g*100:.1f}% of "
              f"{len(acc_pairs)} matched cells (mean Δ = {acc_pairs['d_acc'].mean():+.3f}).")
    n_div_undef = pairs["diversity_greedy"].isna().sum()
    print(f"  Diversity: greedy value undefined (deterministic / too few correct) "
          f"in {n_div_undef} of {len(pairs)} cells.")
    if len(div_pairs):
        frac_c = (div_pairs["d_div"] < 0).mean()
        print(f"            where both defined, categorical > greedy in "
              f"{frac_c*100:.1f}% of {len(div_pairs)} cells "
              f"(mean Δ = {div_pairs['d_div'].mean():+.3f}).")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="Path to the T-sweep results CSV")
    p.add_argument("--outdir", default="figures", help="Directory for output PNGs")
    p.add_argument("--compute-mode", choices=["steps", "mixed", "T"], default="steps",
                   help="Compute axis / budget unit. 'steps' = realised mean steps for all "
                        "decoders (recommended, matches the Q1 wording); 'mixed' = steps for "
                        "EB and requested T for the rest; 'T' = requested T for all.")
    p.add_argument("--xscale", choices=["linear", "log"], default="linear",
                   help="x-axis scale for fig1 (default: linear / uniform)")
    p.add_argument("--raw-diversity", action="store_true",
                   help="Plot raw diversity with per-panel scales instead of the uniform "
                        "min-max-normalised [0,1] scale.")
    p.add_argument("--diversity-index", type=int, default=0,
                   help="Which diversity metric to use for grammars that list several (default: 0)")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    sns.set_theme(style="whitegrid", context="paper")

    df = load_and_prepare(args.file, args.diversity_index)
    df = add_compute_axis(df, args.compute_mode)
    best = select_best_per_budget(df)
    pairs = build_pairs(df)  # Q3 uses ALL cells (fair pairwise comparison), not best-only

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_accuracy_vs_compute(best, outdir / "fig1_accuracy_vs_compute.png",
                            args.dpi, args.xscale, args.compute_mode)
    fig_diversity_vs_accuracy(best, outdir / "fig2_diversity_vs_accuracy.png",
                              args.dpi, normalize=not args.raw_diversity)
    fig_paired_deltas(pairs, outdir / "fig3_paired_deltas.png", args.dpi)
    has_quad = fig_tradeoff_quadrant(pairs, outdir / "fig3b_tradeoff_quadrant.png", args.dpi)

    print(f"Wrote figures to {outdir}/:")
    print("  fig1_accuracy_vs_compute.png")
    print("  fig2_diversity_vs_accuracy.png")
    print("  fig3_paired_deltas.png")
    if has_quad:
        print("  fig3b_tradeoff_quadrant.png")

    print_summary(best, pairs)


if __name__ == "__main__":
    main()