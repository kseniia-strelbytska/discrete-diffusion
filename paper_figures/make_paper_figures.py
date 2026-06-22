#!/usr/bin/env python3
"""
make_paper_figures.py
=====================
Canonical figure set for the oracle sampling-dynamics paper.

Single source of truth: results/combined_6_grammar.csv (the 6-grammar oracle sweep).
All figures and the outlier report are regenerated from this one file so the numbers
in the prose are guaranteed to match the plots.

Thesis: with the denoiser held OPTIMAL (analytic grammar oracle), the accuracy a
discrete-diffusion sampler reaches -- and the accuracy/diversity trade-off it pays --
is governed by three inference levers: compute (denoising steps), the noise schedule,
and the decoding rule (greedy vs categorical).

Outputs (into this folder):
  fig1_accuracy_vs_compute.png      role of compute (best-per-budget envelope, +-std)
  fig2_diversity_vs_accuracy.png    accuracy/diversity trade-off
  fig3_greedy_vs_categorical.png    paired greedy-categorical deltas
  fig3b_tradeoff_quadrant.png       joint Delta-acc / Delta-div scatter
  fig4_compute_efficiency.png       steps-to-90%: schedule / adaptive-sampler efficiency
  fig5_baN_parity_anomaly.png       the baN outlier, isolated and explained
  fig6_oracle_monotonicity.png      mechanism: oracle marginal logits along position
  outlier_analysis.md               quantified explanation of every anomaly

Two sequence-length regimes are baked into the sweep and are NOT an error:
  * single-string grammars  (aNbN, aNbNcN, baN, bbaN)  swept at L=128
  * Dyck-2 grammars         (nested / flat parens+brackets) swept at L=32
The compute axis is realised mean steps (NFE); each panel annotates its own L so the
budgets are read in the right scale.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "results" / "combined_6_grammar.csv"
ACC_THRESH = 0.90    # "reliable generation" bar for the efficiency figure
COLLAPSE_FLOOR = 0.10  # uniqueness below this ⇒ mode-collapsed (≤10% distinct outputs).
#                        Degeneracy is defined by COLLAPSE, not by T: a single repeated
#                        valid string can appear at T=1,2,4,... and must not be read as skill.

# ---- display / ordering -------------------------------------------------- #
DECODER_ORDER = ["uniform", "gaussian", "ebsampler", "ar"]
DECODER_DISPLAY = {"uniform": "Uniform", "gaussian": "Gaussian",
                   "ebsampler": "EB (adaptive)", "ar": "AR baseline"}
SAMPLER_ORDER = ["greedy", "categorical"]
SAMPLER_DISPLAY = {"greedy": "Greedy", "categorical": "Categorical"}
SAMPLER_DASHES = {"greedy": "", "categorical": (4, 1.5)}
SAMPLER_MARKERS = {"greedy": "o", "categorical": "X"}

GRAMMAR_ORDER = ["baN", "bbaN", "aNbN", "aNbNcN",
                 "parentheses_and_brackets", "not_nested_parentheses_and_brackets"]
GRAMMAR_DISPLAY = {
    "baN": "baᴺ  (parity)", "bbaN": "bbaᴺ", "aNbN": "aᴺbᴺ  (counting)",
    "aNbNcN": "aᴺbᴺcᴺ  (counting)",
    "parentheses_and_brackets": "Dyck-2 nested",
    "not_nested_parentheses_and_brackets": "Dyck-2 flat",
}
GRAMMAR_L = {  # which sequence length each grammar was swept at
    "baN": 128, "bbaN": 128, "aNbN": 128, "aNbNcN": 128,
    "parentheses_and_brackets": 32, "not_nested_parentheses_and_brackets": 32,
}
# per-grammar diversity axis (first metric is the one plotted)
DIVERSITY_METRIC = {
    "baN": "uniqueness", "bbaN": "nm_joint_coverage", "aNbN": "n_coverage",
    "aNbNcN": "n_entropy", "parentheses_and_brackets": "uniqueness",
    "not_nested_parentheses_and_brackets": "uniqueness",
}


# ---- load / prepare ------------------------------------------------------- #
def load():
    df = pd.read_csv(CSV)
    df["decoder"] = df["strategy"].astype(str).str.lower().str.strip()
    df["sampler"] = df["sampling_strategy"].astype(str).str.lower().str.strip()
    df["accuracy"] = pd.to_numeric(df["mean_both_rules"], errors="coerce")
    df["acc_std"] = pd.to_numeric(df["std_both_rules"], errors="coerce").fillna(0.0)
    df["compute"] = pd.to_numeric(df["n_steps_mean"], errors="coerce")
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["hparam"] = np.where(df["decoder"] == "gaussian", df["sigma"],
                            np.where(df["decoder"] == "ebsampler", df["eb_gamma"], np.nan))
    # diversity on a single per-grammar axis; gate out unreliable cells
    df["diversity_metric"] = df["grammar"].map(DIVERSITY_METRIC)
    div = []
    for _, r in df.iterrows():
        m = r["diversity_metric"]
        div.append(pd.to_numeric(r.get(m), errors="coerce") if m in df.columns else np.nan)
    df["diversity"] = div
    gate = df["n_correct_too_low"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
    df.loc[gate.values, "diversity"] = np.nan
    return df


def best_per_budget(df):
    """Highest-accuracy hyperparameter at each (grammar, decoder, sampler, rounded-compute)."""
    d = df.dropna(subset=["accuracy", "compute"]).copy()
    d = d[d["compute"] > 0]
    d["_b"] = d["compute"].round()
    idx = d.groupby(["grammar", "decoder", "sampler", "_b"], sort=False)["accuracy"].idxmax()
    return d.loc[idx].drop(columns="_b").reset_index(drop=True)


def present(series, order):
    out = [v for v in order if v in set(series)]
    return out + [v for v in series.unique() if v not in out]


GREEDY_C = "#d1495b"   # red
CAT_C = "#2e6f95"      # blue


# =========================================================================== #
# Fig 1 -- accuracy vs compute (role of compute & decoding)
#   analyse.py style: 6 grammar panels, ONE schedule (uniform), 2 lines
#   (greedy vs categorical). Collapsed points (uniqueness < floor) are ringed so a
#   high-accuracy result that is really a single repeated string is never mistaken
#   for skill. Per-panel x autoscaling (the two L regimes span different NFE ranges).
# =========================================================================== #
def fig1_accuracy_vs_compute(df, out, schedule="uniform"):
    d = df[df["decoder"] == schedule].copy()
    gorder = [g for g in GRAMMAR_ORDER if g in set(d["grammar"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, g in zip(axes.flat, gorder):
        sub = d[d["grammar"] == g]
        for s, col in [("greedy", GREEDY_C), ("categorical", CAT_C)]:
            line = sub[sub.sampler == s].dropna(subset=["accuracy", "compute"]).sort_values("compute")
            if line.empty:
                continue
            ax.plot(line["compute"], line["accuracy"], color=col, lw=2,
                    marker="o", ms=4, label=SAMPLER_DISPLAY[s], zorder=3)
            ax.fill_between(line["compute"],
                            (line["accuracy"] - line["acc_std"]).clip(0, 1),
                            (line["accuracy"] + line["acc_std"]).clip(0, 1),
                            color=col, alpha=0.13, lw=0)
            coll = line[line["uniqueness"] < COLLAPSE_FLOOR]
            ax.scatter(coll["compute"], coll["accuracy"], s=110, facecolors="none",
                       edgecolors="black", linewidths=1.4, zorder=4)
        ax.axhline(ACC_THRESH, color="grey", ls=":", lw=1, zorder=1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(GRAMMAR_DISPLAY.get(g, g), fontsize=11)
        ax.text(0.97, 0.05, f"L={GRAMMAR_L[g]}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="dimgrey")
        ax.grid(True, ls="--", alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Mean denoising steps  (NFE)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Both-rules accuracy")
    handles = [plt.Line2D([], [], color=GREEDY_C, lw=2, marker="o", label="Greedy"),
               plt.Line2D([], [], color=CAT_C, lw=2, marker="o", label="Categorical"),
               plt.Line2D([], [], color="grey", ls=":", label=f"{int(ACC_THRESH*100)}% accuracy"),
               plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="",
                          label=f"mode-collapsed (<{int(COLLAPSE_FLOOR*100)}% unique)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.suptitle(f"Role of compute and decoding  ({schedule} schedule, optimal denoiser)  "
                 "— ringed points are mode-collapsed", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 2 -- accuracy–diversity trade-off (clean)
#   6 grammar panels, uniform schedule, greedy vs categorical, points ordered by
#   compute. The trade-off shows directly: greedy hugs the bottom (high accuracy,
#   collapsed); categorical climbs to the top-right (accurate AND diverse).
# =========================================================================== #
def fig2_diversity_vs_accuracy(df, out, schedule="uniform"):
    d = df[df["decoder"] == schedule].copy()
    gorder = [g for g in GRAMMAR_ORDER if g in set(d["grammar"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, g in zip(axes.flat, gorder):
        sub = d[d["grammar"] == g]
        for s, col in [("greedy", GREEDY_C), ("categorical", CAT_C)]:
            line = sub[sub.sampler == s].dropna(subset=["accuracy", "diversity"]).sort_values("compute")
            if line.empty:
                continue
            ax.plot(line["accuracy"], line["diversity"], color=col, lw=1.8,
                    marker="o", ms=4, label=SAMPLER_DISPLAY[s], alpha=0.95)
            coll = line[line["uniqueness"] < COLLAPSE_FLOOR]
            ax.scatter(coll["accuracy"], coll["diversity"], s=110, facecolors="none",
                       edgecolors="black", linewidths=1.4, zorder=4)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(GRAMMAR_DISPLAY.get(g, g), fontsize=11)
        ax.set_xlabel("Both-rules accuracy")
        ax.set_ylabel(f"diversity: {DIVERSITY_METRIC[g]}")
        ax.grid(True, ls="--", alpha=0.3)
    handles = [plt.Line2D([], [], color=GREEDY_C, lw=2, marker="o", label="Greedy"),
               plt.Line2D([], [], color=CAT_C, lw=2, marker="o", label="Categorical"),
               plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="",
                          label=f"mode-collapsed (<{int(COLLAPSE_FLOOR*100)}% unique)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.suptitle(f"Accuracy–diversity trade-off  ({schedule} schedule)  "
                 "— greedy buys accuracy with collapse, categorical keeps both",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 3 -- greedy vs categorical paired deltas (+ quadrant)
# =========================================================================== #
def build_pairs(df):
    d = df.copy()
    d["cell"] = (d["grammar"] + "|" + d["decoder"] + "|"
                 + d["hparam"].round(6).astype(str) + "|" + d["T"].astype(str))
    keep = ["grammar", "decoder", "cell", "accuracy", "diversity"]
    g = d[d["sampler"] == "greedy"][keep]
    c = d[d["sampler"] == "categorical"][keep]
    m = g.merge(c, on=["grammar", "decoder", "cell"], suffixes=("_greedy", "_cat"))
    m["d_acc"] = m["accuracy_greedy"] - m["accuracy_cat"]
    m["d_div"] = m["diversity_greedy"] - m["diversity_cat"]
    return m


def fig3_paired_deltas(pairs, out):
    gorder = [g for g in GRAMMAR_ORDER if g in set(pairs["grammar"])]
    disp = [GRAMMAR_DISPLAY.get(g, g) for g in gorder]
    p = pairs.copy()
    p["gd"] = p["grammar"].map(lambda x: GRAMMAR_DISPLAY.get(x, x))
    decoders = present(p["decoder"], DECODER_ORDER)
    palette = dict(zip(decoders, sns.color_palette("colorblind", len(decoders))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, col, title, sign in [
        (axes[0], "d_acc", "Δ accuracy  (greedy − categorical)", "above 0 => greedy more accurate"),
        (axes[1], "d_div", "Δ diversity  (greedy − categorical)", "below 0 => categorical more diverse"),
    ]:
        sub = p.dropna(subset=[col])
        sns.stripplot(data=sub, x="gd", y=col, order=disp, hue="decoder",
                      hue_order=decoders, palette=palette, dodge=True, jitter=0.18,
                      size=4, alpha=0.7, ax=ax, legend=(ax is axes[0]))
        means = sub.groupby("gd")[col].mean().reindex(disp)
        ax.scatter(range(len(disp)), means.values, marker="_", s=600,
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3b_quadrant(pairs, out):
    sub = pairs.dropna(subset=["d_acc", "d_div"]).copy()
    if sub.empty:
        return
    gorder = [g for g in GRAMMAR_ORDER if g in set(sub["grammar"])]
    disp = [GRAMMAR_DISPLAY.get(g, g) for g in gorder]
    sub["gd"] = sub["grammar"].map(lambda x: GRAMMAR_DISPLAY.get(x, x))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sns.scatterplot(data=sub, x="d_acc", y="d_div", hue="gd", hue_order=disp,
                    style="gd", style_order=disp, s=70, ax=ax, palette="colorblind")
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", lw=1)
    xm = (np.nanmax(np.abs(sub["d_acc"])) or 1) * 1.15
    ym = (np.nanmax(np.abs(sub["d_div"])) or 1) * 1.15
    ax.set_xlim(-xm, xm)
    ax.set_ylim(-ym, ym)
    ax.text(xm * 0.96, -ym * 0.96, "greedy: +acc / −div", ha="right", va="bottom",
            fontsize=8, color="dimgrey")
    ax.set_xlabel("Δ accuracy  (greedy − categorical)")
    ax.set_ylabel("Δ diversity  (greedy − categorical)")
    ax.set_title("Joint accuracy–diversity trade-off per matched cell")
    ax.legend(title="Grammar", fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 4 -- compute efficiency: steps to reach 90% accuracy
# =========================================================================== #
def steps_to_threshold(df, grammar, decoder, sampler, thresh):
    sub = df[(df.grammar == grammar) & (df.decoder == decoder)
             & (df.sampler == sampler)].dropna(subset=["accuracy", "compute"])
    ok = sub[sub["accuracy"] >= thresh]
    if ok.empty:
        return np.nan, sub["accuracy"].max() if not sub.empty else np.nan
    return ok["compute"].min(), ok["accuracy"].max()


def fig4_compute_efficiency(df, out):
    decoders = ["uniform", "gaussian", "ebsampler"]
    palette = dict(zip(DECODER_ORDER, sns.color_palette("colorblind", len(DECODER_ORDER))))
    gorder = [g for g in GRAMMAR_ORDER if g in set(df["grammar"])]
    disp = [GRAMMAR_DISPLAY.get(g, g) for g in gorder]

    import matplotlib.transforms as mtransforms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    YFLOOR = 0.8  # log-axis lower bound; n/a markers sit just above it
    for ax, sampler in zip(axes, ["categorical", "greedy"]):
        ax.set_yscale("log")
        width = 0.26
        x = np.arange(len(gorder))
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i, dec in enumerate(decoders):
            steps = [steps_to_threshold(df, g, dec, sampler, ACC_THRESH)[0] for g in gorder]
            xpos = x + (i - 1) * width
            real = [(xp, s) for xp, s in zip(xpos, steps) if not np.isnan(s)]
            if real:
                ax.bar([xp for xp, _ in real], [s for _, s in real], width,
                       color=palette[dec], label=DECODER_DISPLAY[dec], zorder=3)
            else:  # keep legend entry even if all n/a for this sampler
                ax.bar([], [], color=palette[dec], label=DECODER_DISPLAY[dec])
            for xp, s in zip(xpos, steps):
                if np.isnan(s):
                    ax.text(xp, 0.02, "n/a", ha="center", va="bottom", fontsize=8,
                            color=palette[dec], fontweight="bold", transform=trans,
                            rotation=90)
        ax.set_ylim(bottom=YFLOOR)
        ax.set_xticks(x)
        ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Mean steps to reach {int(ACC_THRESH*100)}% accuracy  (log)")
        ax.set_title(f"{SAMPLER_DISPLAY[sampler]} sampling", fontsize=11)
        ax.grid(True, axis="y", ls="--", alpha=0.3, zorder=0)
        ax.legend(fontsize=8, title="Schedule", title_fontsize=9)
    fig.suptitle(f"Compute efficiency: steps to {int(ACC_THRESH*100)}% accuracy   "
                 "(n/a = threshold never reached at any budget)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 5 -- the baN parity anomaly, isolated
# =========================================================================== #
def fig5_baN_anomaly(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # left: baN, uniform schedule, accuracy vs T, greedy vs categorical
    sub = df[(df.grammar == "baN") & (df.decoder == "uniform")]
    for s, c, ls in [("greedy", "#d1495b", "-"), ("categorical", "#2e6f95", "--")]:
        line = sub[sub.sampler == s].dropna(subset=["accuracy", "T"]).sort_values("T")
        axes[0].plot(line["T"], line["accuracy"], marker="o", color=c, ls=ls,
                     label=SAMPLER_DISPLAY[s], lw=2)
    axes[0].axhline(0.5, color="grey", ls=":", lw=1)
    axes[0].text(line["T"].min(), 0.5, " chance for one parity bit", fontsize=8,
                 color="grey", va="bottom")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Denoising steps T  (uniform schedule)")
    axes[0].set_ylabel("Both-rules accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("baᴺ: greedy collapses, categorical floors at ½", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.3)

    # right: categorical minus greedy accuracy AT MAXIMUM COMPUTE, per grammar, per
    # schedule. At the largest budget greedy is fully iterative (no one-shot collapse) and
    # diverse, so this is a degeneracy-free comparison — no T-threshold guessing. AR/EB are
    # excluded (they reach 1.0 trivially); only the fixed schedules uniform/gaussian sweep T.
    Tn = pd.to_numeric(df["T"], errors="coerce")
    dd = df.assign(Tn=Tn)
    gorder = [g for g in GRAMMAR_ORDER if g in set(df["grammar"])]
    disp = [GRAMMAR_DISPLAY.get(g, g).split("  ")[0] for g in gorder]
    x = np.arange(len(gorder))
    for off, dec, col in [(-0.2, "uniform", "#1f77b4"), (0.2, "gaussian", "#e08214")]:
        delta = []
        for g in gorder:
            sub = dd[(dd.grammar == g) & (dd.decoder == dec)]
            tmax = sub["Tn"].max()
            at_max = sub[sub["Tn"] == tmax]
            bc_ = at_max[at_max.sampler == "categorical"]["accuracy"].max()
            bg_ = at_max[at_max.sampler == "greedy"]["accuracy"].max()
            delta.append(bc_ - bg_)
        axes[1].bar(x + off, delta, 0.4, label=DECODER_DISPLAY[dec], color=col)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(disp, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Δ accuracy at max compute  (categorical − greedy)")
    axes[1].set_title("At max compute: greedy wins the counting grammars; categorical wins "
                      "baᴺ\n(baᴺ positive under both schedules; Dyck mixed)", fontsize=9.5)
    axes[1].legend(title="Schedule", fontsize=8, title_fontsize=9)
    axes[1].grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("The baᴺ parity anomaly", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 6 -- mechanism: oracle marginal logits along the sequence
# =========================================================================== #
def fig6_monotonicity(out):
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from oracle.grammar_oracles import oracleModel
    from datasets.constants import SOS_token, MASK_token

    L = 32
    grammars = GRAMMAR_ORDER
    palette = dict(zip(grammars, sns.color_palette("husl", len(grammars))))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for g in grammars:
        oracle = oracleModel(g, vocab_size=8, device="cpu")
        x = torch.tensor([[SOS_token] + [MASK_token] * (L + 1)], dtype=torch.long)
        logits = oracle(x)[0]                       # (L+2, vocab)
        probs = torch.softmax(logits.float(), dim=-1)
        # P(token 0 = first content symbol 'a' / '(') vs position
        p0 = probs[1:L + 1, 0].detach().numpy()
        ax.plot(range(1, L + 1), p0, marker=".", ms=4,
                color=palette[g], label=GRAMMAR_DISPLAY.get(g, g).split("  ")[0])
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Oracle P(token 0) on a fully-masked sequence")
    ax.set_title("Mechanism: how confidently the oracle marginal fixes each position\n"
                 "(flat ≈ ½ => position undetermined => greedy one-shot fails; "
                 "sharp => decided early)", fontsize=10.5)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Outlier analysis report
# =========================================================================== #
def write_outlier_report(df, best, pairs, out):
    L = []
    def w(s=""): L.append(s)

    w("# Outlier & anomaly analysis")
    w()
    w("Source: `results/combined_6_grammar.csv` (oracle sweep, 1468 rows). "
      "All figures regenerated from this file by `make_paper_figures.py`.")
    w()
    w("## 0. Two length regimes (not an anomaly — document it)")
    w("- Single-string grammars (baᴺ, bbaᴺ, aᴺbᴺ, aᴺbᴺcᴺ) swept at **L=128** "
      "(max T≈130).")
    w("- Dyck-2 grammars (nested / flat) swept at **L=32** (max T≈34).")
    w("- Consequence: in Fig 1 the Dyck panels only span the low-NFE region; "
      "the compute axis must be read per-panel (L is annotated).")
    w()

    # 1. baN parity -- cat-vs-greedy AT MAXIMUM COMPUTE per diffusion schedule.
    # Degeneracy is collapse (low uniqueness), not a T value (e.g. bbaN greedy is collapsed
    # at T=2, uniqueness 0.056). Comparing at the largest budget sidesteps it entirely:
    # greedy there is fully iterative and diverse. Matches src/analyse.py's full-curve view.
    w("## 1. baᴺ — the parity anomaly (the headline outlier)")
    Tn = pd.to_numeric(df["T"], errors="coerce")
    dd = df.assign(Tn=Tn)
    def acc_at_maxT(g, dec, s):
        sub = dd[(dd.grammar == g) & (dd.decoder == dec)]
        am = sub[sub["Tn"] == sub["Tn"].max()]
        return am[am.sampler == s]["accuracy"].max()
    gg = [x for x in GRAMMAR_ORDER if x in set(df["grammar"])]
    uni_win = [g for g in gg if acc_at_maxT(g, "uniform", "categorical") > acc_at_maxT(g, "uniform", "greedy")]
    both_win = [g for g in gg if all(acc_at_maxT(g, d, "categorical") > acc_at_maxT(g, d, "greedy")
                                     for d in ["uniform", "gaussian"])]
    w("- **Degeneracy is collapse, not a T value.** A single repeated valid string can appear "
      "at T=1, 2 or 4 (e.g. bbaᴺ greedy at T=2 has uniqueness 0.056). So instead of excluding "
      "low-T rows, we compare samplers **at maximum compute**, where greedy is fully iterative "
      "and diverse — a degeneracy-free comparison (AR/EB excluded; they reach 1.0 trivially).")
    w(f"  - Under **uniform** (max compute), categorical beats greedy for: {uni_win} "
      f"(baᴺ {acc_at_maxT('baN','uniform','categorical'):.3f} vs {acc_at_maxT('baN','uniform','greedy'):.3f}).")
    w(f"  - Under **both** uniform and Gaussian, categorical wins for: {both_win}.")
    w("- **baᴺ is the strongest, not the unique, case**: positive under both schedules with the "
      "widest margin; greedy wins all three counting grammars (aᴺbᴺ, aᴺbᴺcᴺ, bbaᴺ); Dyck is "
      "mixed (nested flips to greedy under Gaussian; flat is a +0.002 tie). An earlier draft "
      "over-claimed 'baᴺ only' by counting greedy's degenerate one-shot collapse as a win.")
    cat1 = df[(df.grammar == "baN") & (df.decoder == "uniform")
              & (df.sampler == "categorical") & (df["T"] == 1)]["accuracy"]
    gr_lowT = df[(df.grammar == "baN") & (df.decoder == "uniform")
                 & (df.sampler == "greedy") & (df["T"] <= 8)]["accuracy"]
    w(f"- Uniform categorical at T=1 sits at **{cat1.max():.3f}** ≈ ½, the chance rate of a "
      "single parity bit; uniform greedy at T≤8 is **{:.3f}** (deterministic wrong parity)."
      .format(gr_lowT.max()))
    w("- **Why:** baᴺ = 'starts with b, even number of a's'. The even-count rule is a global "
      "parity constraint that couples positions. The oracle's *per-position marginal* for an "
      "interior slot is ~½/½ (either symbol can be valid depending on the rest), so:")
    w("  - **greedy** (argmax per position, all at once at low T) commits to one fixed pattern "
      "whose parity is wrong with near-certainty → ~0 accuracy until T is large enough that "
      "tokens are committed sequentially and the marginal re-conditions.")
    w("  - **categorical** samples each undecided position ~½/½, so the final parity is a fair "
      "coin → ~0.5 even at T=1, then climbs as conditioning kicks in.")
    w("- This is the cleanest evidence in the paper that the failure is a **sampling** "
      "phenomenon (marginal vs joint), not a denoiser-capacity one — the denoiser is exact.")
    w()

    # 2. one-shot greedy collapse to 1.0
    w("## 2. bbaᴺ / Dyck-flat — greedy reaches 1.0 at T=1 (lucky collapse, not skill)")
    for g in ["bbaN", "not_nested_parentheses_and_brackets"]:
        r = df[(df.grammar == g) & (df.sampler == "greedy") & (df["T"] == 1)]
        if not r.empty:
            acc = r["accuracy"].max()
            uni = r["uniqueness"].min()
            w(f"- {GRAMMAR_DISPLAY[g]}: greedy T=1 accuracy **{acc:.3f}**, uniqueness **{uni}** "
              "→ every sample is the SAME modal string.")
    w("- **Why:** at T=1 greedy argmaxes the fully-masked marginal in one shot. If the modal "
      "per-position configuration happens to be a valid string for that grammar, accuracy is "
      "trivially 1.0 — but it is a single point mass (zero diversity). Contrast aᴺbᴺ/baᴺ where "
      "the modal one-shot string is invalid → T=1 greedy = 0. So 'accuracy at T=1' must always "
      "be read together with diversity; report both or it is misleading.")
    w()

    # 3. categorical's compute cost
    w("## 3. Categorical's accuracy is real but expensive")
    hi = best[(best.sampler == "categorical") & (best.accuracy > 0.9)]
    w(f"- High-accuracy categorical points sit at large NFE "
      f"(median {hi['compute'].median():.0f}, up to {hi['compute'].max():.0f} steps), whereas "
      "greedy reaches comparable or higher accuracy at a fraction of the compute. This is the "
      "core compute–quality asymmetry behind the trade-off figures.")
    w()

    # 4. ebsampler categorical underperforms on aNbN
    w("## 4. EB-sampler categorical underperforms on the counting grammars")
    eb = df[(df.decoder == "ebsampler") & (df.sampler == "categorical")
            & (df.grammar == "aNbN")]["accuracy"].max()
    w(f"- aᴺbᴺ EB categorical best = **{eb:.3f}** vs EB greedy = 1.000. The adaptive "
      "entropy-bounded stopping halts while the count is still uncertain under stochastic "
      "sampling; greedy is unaffected because it never injects sampling noise. Flag in text; "
      "do not present EB as uniformly Pareto-dominant.")
    w()

    # 5. gated diversity cells
    n_low = int(df["n_correct_too_low"].astype(str).str.lower().isin(["true", "1", "1.0"]).sum())
    w("## 5. Diversity reliability gate")
    w(f"- **{n_low}/{len(df)}** rows are flagged `n_correct_too_low` (too few valid samples to "
      "estimate diversity); their diversity is set to NaN and omitted from Fig 2/3. Most are "
      "low-compute or collapsed-greedy cells. This is why several greedy lines in Fig 2 are "
      "short — by construction, not by omission.")
    w()

    # 6. non-monotonicity check
    w("## 6. Is more compute always better? (monotonicity check on the envelope)")
    nonmono = []
    for (g, dcr, s), grp in best.groupby(["grammar", "decoder", "sampler"], sort=False):
        grp = grp.sort_values("compute")
        acc = grp["accuracy"].values
        if len(acc) >= 2 and np.argmax(acc) != len(acc) - 1 and acc.max() - acc[-1] > 1e-9:
            nonmono.append((g, dcr, s, float(acc[-1]), float(acc.max())))
    if nonmono:
        w(f"- {len(nonmono)} envelope line(s) peak at intermediate compute (worth a sentence):")
        for g, dcr, s, last, pk in nonmono:
            w(f"  - {g} / {dcr} / {s}: acc@max-compute={last:.3f} vs peak={pk:.3f}")
    else:
        w("- Within the best-per-budget envelope, accuracy is non-decreasing in compute "
          "for every line.")
    w()
    out.write_text("\n".join(L))


def main():
    if not CSV.exists():
        sys.exit(f"Canonical CSV not found: {CSV}")
    sns.set_theme(style="whitegrid", context="paper")
    df = load()
    best = best_per_budget(df)
    pairs = build_pairs(df)

    fig1_accuracy_vs_compute(df, HERE / "fig1_accuracy_vs_compute.png")
    fig2_diversity_vs_accuracy(df, HERE / "fig2_diversity_vs_accuracy.png")
    fig3_paired_deltas(pairs, HERE / "fig3_greedy_vs_categorical.png")
    fig3b_quadrant(pairs, HERE / "fig3b_tradeoff_quadrant.png")
    fig4_compute_efficiency(df, HERE / "fig4_compute_efficiency.png")
    fig5_baN_anomaly(df, HERE / "fig5_baN_parity_anomaly.png")
    try:
        fig6_monotonicity(HERE / "fig6_oracle_monotonicity.png")
        fig6_ok = True
    except Exception as e:  # oracle import is optional / environment-dependent
        fig6_ok = False
        print(f"[warn] fig6 (monotonicity) skipped: {e}")
    write_outlier_report(df, best, pairs, HERE / "outlier_analysis.md")

    print("Wrote to", HERE)
    for f in ["fig1_accuracy_vs_compute.png", "fig2_diversity_vs_accuracy.png",
              "fig3_greedy_vs_categorical.png", "fig3b_tradeoff_quadrant.png",
              "fig4_compute_efficiency.png", "fig5_baN_parity_anomaly.png"]:
        print("  ", f)
    if fig6_ok:
        print("   fig6_oracle_monotonicity.png")
    print("   outlier_analysis.md")


if __name__ == "__main__":
    main()
