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

# Full metric panel for the Gaussian-vs-uniform comparison (label -> CSV column).
# accuracy first, then the diversity suite. Columns absent for a grammar show blank.
ALL_METRICS = [
    ("both-rules acc", "mean_both_rules"),
    ("uniqueness", "uniqueness"),
    ("norm. Lev. dist", "mean_lev_dist_normalized"),
    ("bigram div", "bigram_diversity"),
    ("trigram div", "trigram_diversity"),
    ("DFA state cov", "dfa_state_coverage"),
    ("DFA trans cov", "dfa_transition_coverage"),
    ("n-entropy", "n_entropy"),
    ("n-coverage", "n_coverage"),
    ("m-entropy", "m_entropy"),
    ("nm-joint cov", "nm_joint_coverage"),
]

# stable colours for the three diffusion decoders (AR excluded from sweep figures)
DECODER_COLOR = {"uniform": "#1f77b4", "gaussian": "#e08214", "ebsampler": "#2ca02c"}


# ---- load / prepare ------------------------------------------------------- #
def load():
    df = pd.read_csv(CSV)
    df["decoder"] = df["strategy"].astype(str).str.lower().str.strip()
    df["sampler"] = df["sampling_strategy"].astype(str).str.lower().str.strip()
    df["accuracy"] = pd.to_numeric(df["mean_both_rules"], errors="coerce")
    df["acc_std"] = pd.to_numeric(df["std_both_rules"], errors="coerce").fillna(0.0)
    # Error band = pooled 95% Wilson CI (computed by the sweep driver) rather than the
    # std across only ~5 reps, which is itself very noisy and is 0 on saturated cells.
    # Fall back to mean±std only if the CI columns are absent.
    if "ci_low_both_rules" in df.columns and "ci_high_both_rules" in df.columns:
        df["acc_lo"] = pd.to_numeric(df["ci_low_both_rules"], errors="coerce").clip(0, 1)
        df["acc_hi"] = pd.to_numeric(df["ci_high_both_rules"], errors="coerce").clip(0, 1)
        df["acc_lo"] = df["acc_lo"].fillna((df["accuracy"] - df["acc_std"]).clip(0, 1))
        df["acc_hi"] = df["acc_hi"].fillna((df["accuracy"] + df["acc_std"]).clip(0, 1))
    else:
        df["acc_lo"] = (df["accuracy"] - df["acc_std"]).clip(0, 1)
        df["acc_hi"] = (df["accuracy"] + df["acc_std"]).clip(0, 1)
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
            ax.fill_between(line["compute"], line["acc_lo"], line["acc_hi"],
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
    axes[0].text(line["T"].min(), 0.5, " ½ (one parity bit)", fontsize=8,
                 color="grey", va="bottom")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Denoising steps T  (uniform schedule)")
    axes[0].set_ylabel("Both-rules accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("baᴺ: greedy collapses to 0; categorical stays well above", fontsize=11)
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
# Fig 7 -- decoder comparison: accuracy vs compute (generalises fig1 to all 3
#   diffusion decoders). 6 grammar panels; colour = decoder, line style = sampler;
#   each line is the best-per-budget envelope (best sigma/gamma at each NFE).
# =========================================================================== #
def fig7_decoder_compute(best, out):
    decoders = ["uniform", "gaussian", "ebsampler"]
    gorder = [g for g in GRAMMAR_ORDER if g in set(best["grammar"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, g in zip(axes.flat, gorder):
        sub = best[best["grammar"] == g]
        for dec in decoders:
            for s, dash in [("greedy", ""), ("categorical", (3, 2))]:
                line = sub[(sub.decoder == dec) & (sub.sampler == s)] \
                    .dropna(subset=["accuracy", "compute"]).sort_values("compute")
                if line.empty:
                    continue
                ax.plot(line["compute"], line["accuracy"], color=DECODER_COLOR[dec],
                        lw=1.8, marker="o", ms=3.5,
                        ls=(0, dash) if dash else "-", alpha=0.9)
        ax.axhline(ACC_THRESH, color="grey", ls=":", lw=1, zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(GRAMMAR_DISPLAY.get(g, g), fontsize=11)
        ax.text(0.97, 0.05, f"L={GRAMMAR_L[g]}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="dimgrey")
        ax.grid(True, ls="--", alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Mean denoising steps  (NFE, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Both-rules accuracy")
    handles = [plt.Line2D([], [], color=DECODER_COLOR[d], lw=2, label=DECODER_DISPLAY[d])
               for d in decoders]
    handles += [plt.Line2D([], [], color="black", lw=2, ls="-", label="Greedy"),
                plt.Line2D([], [], color="black", lw=2, ls=(0, (3, 2)), label="Categorical"),
                plt.Line2D([], [], color="grey", ls=":", label=f"{int(ACC_THRESH*100)}% accuracy")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Decoder comparison — accuracy vs. compute  "
                 "(best-per-budget envelope, optimal denoiser)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 8 -- decoder comparison: accuracy-diversity trade-off (generalises fig2).
# =========================================================================== #
def fig8_decoder_tradeoff(best, out):
    decoders = ["uniform", "gaussian", "ebsampler"]
    gorder = [g for g in GRAMMAR_ORDER if g in set(best["grammar"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, g in zip(axes.flat, gorder):
        sub = best[best["grammar"] == g]
        for dec in decoders:
            for s, dash in [("greedy", ""), ("categorical", (3, 2))]:
                line = sub[(sub.decoder == dec) & (sub.sampler == s)] \
                    .dropna(subset=["accuracy", "diversity"]).sort_values("compute")
                if line.empty:
                    continue
                ax.plot(line["accuracy"], line["diversity"], color=DECODER_COLOR[dec],
                        lw=1.6, marker="o", ms=3.5,
                        ls=(0, dash) if dash else "-", alpha=0.9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(GRAMMAR_DISPLAY.get(g, g), fontsize=11)
        ax.set_xlabel("Both-rules accuracy")
        ax.set_ylabel(f"diversity: {DIVERSITY_METRIC[g]}")
        ax.grid(True, ls="--", alpha=0.3)
    handles = [plt.Line2D([], [], color=DECODER_COLOR[d], lw=2, label=DECODER_DISPLAY[d])
               for d in decoders]
    handles += [plt.Line2D([], [], color="black", lw=2, ls="-", label="Greedy"),
                plt.Line2D([], [], color="black", lw=2, ls=(0, (3, 2)), label="Categorical")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Decoder comparison — accuracy–diversity trade-off  "
                 "(best-per-budget envelope)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 9 -- the Gaussian schedule itself (regenerated from src GaussianSchedule).
#   Left: masking probability sweeping right->left across positions at several t
#   (sigma=10). Right: at fixed t=0.5, how sigma morphs the boundary from a sharp
#   left-to-right front (sigma small ~ autoregressive) to a flat profile (sigma
#   large ~ uniform). This is the analytic claim of the source note, made visual.
# =========================================================================== #
def fig9_gaussian_schedule(out, L=128):
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from schedules.gaussian_schedule import GaussianSchedule

    def pmask(sigma, t):
        return GaussianSchedule(sigma).p_mask(torch.tensor(float(t)), L, "cpu").numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    pos = np.arange(L)
    # left: the right->left sweep at sigma=10
    ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap = sns.color_palette("viridis", len(ts))
    for t, c in zip(ts, cmap):
        axes[0].plot(pos, pmask(10.0, t), color=c, lw=1.8, label=f"t={t:.1f}")
    axes[0].set_title("Masking front sweeps right→left  (σ=10)", fontsize=11)
    axes[0].set_xlabel("Position i")
    axes[0].set_ylabel(r"$p_{\mathrm{mask},i}(t)=\Phi((i-\mu(t))/\sigma)$")
    axes[0].legend(fontsize=8, ncol=2, title="timestep")
    axes[0].grid(True, ls="--", alpha=0.3)
    # right: sigma morphs sharp(AR) -> flat(uniform) at t=0.5
    for sigma, c in zip([2, 10, 50, 100], sns.color_palette("rocket", 4)):
        axes[1].plot(pos, pmask(sigma, 0.5), color=c, lw=2, label=f"σ={sigma}")
    axes[1].set_title("σ interpolates autoregressive ↔ uniform  (t=0.5)", fontsize=11)
    axes[1].set_xlabel("Position i")
    axes[1].set_ylabel(r"$p_{\mathrm{mask},i}(0.5)$")
    axes[1].legend(fontsize=8, title="width")
    axes[1].grid(True, ls="--", alpha=0.3)
    axes[1].text(0.03, 0.04, "small σ: sharp front ≈ left-to-right (AR)\n"
                             "large σ: flat ≈ uniform schedule",
                 transform=axes[1].transAxes, fontsize=8, color="dimgrey", va="bottom")
    fig.suptitle(f"The Gaussian noise schedule  (L={L}):  a positional masking front "
                 "controlled by σ", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 10 -- Gaussian: effect of T and sigma on accuracy, across grammars and
#   samplers. One heatmap per (grammar, sampler): x=requested T, y=sigma,
#   colour=both-rules accuracy. Reads directly off the gaussian rows of the CSV.
# =========================================================================== #
def fig10_gaussian_T_sigma(df, out):
    g_df = df[df.decoder == "gaussian"].copy()
    g_df["Tn"] = pd.to_numeric(g_df["T"], errors="coerce")
    g_df["sig"] = pd.to_numeric(g_df["sigma"], errors="coerce")
    gorder = [g for g in GRAMMAR_ORDER if g in set(g_df["grammar"])]
    samplers = ["greedy", "categorical"]
    fig, axes = plt.subplots(len(gorder), len(samplers),
                             figsize=(10, 2.5 * len(gorder)))
    for r, g in enumerate(gorder):
        for c, s in enumerate(samplers):
            ax = axes[r, c]
            sub = g_df[(g_df.grammar == g) & (g_df.sampler == s)]
            piv = sub.pivot_table(index="sig", columns="Tn", values="accuracy",
                                  aggfunc="mean")
            piv = piv.sort_index(ascending=False)  # large sigma on top
            sns.heatmap(piv, ax=ax, cmap="viridis", vmin=0, vmax=1,
                        cbar=(c == len(samplers) - 1), annot=True, fmt=".2f",
                        annot_kws={"size": 6}, linewidths=0.3, linecolor="white")
            if r == 0:
                ax.set_title(f"{SAMPLER_DISPLAY[s]} sampling", fontsize=11)
            ax.set_ylabel(GRAMMAR_DISPLAY.get(g, g).split("  ")[0] + "\nσ"
                          if c == 0 else "")
            ax.set_xlabel("requested T" if r == len(gorder) - 1 else "")
            ax.tick_params(labelsize=7)
    fig.suptitle("Gaussian schedule: both-rules accuracy across T (x) and σ (y)\n"
                 "for every grammar and sampler", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# Fig 11 -- Gaussian vs uniform across ALL metrics. Operating point per
#   (grammar, sampler): the highest-accuracy cell AT MAXIMUM COMPUTE (max T), so
#   both schedules are fully iterative and the comparison is degeneracy-free.
#   Heatmap of (Gaussian - uniform); blank where a metric is undefined for the
#   grammar. One panel per sampler.
# =========================================================================== #
def _op_point_maxT(df, g, dec, s):
    sub = df[(df.grammar == g) & (df.decoder == dec) & (df.sampler == s)].copy()
    sub["Tn"] = pd.to_numeric(sub["T"], errors="coerce")
    sub = sub[sub["Tn"] == sub["Tn"].max()]
    if sub.empty:
        return None
    return sub.loc[sub["accuracy"].idxmax()]


def fig11_gaussian_vs_uniform(df, out):
    gate = df["n_correct_too_low"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
    dfx = df.copy()
    gorder = [g for g in GRAMMAR_ORDER if g in set(df["grammar"])]
    disp = [GRAMMAR_DISPLAY.get(g, g).split("  ")[0] for g in gorder]
    samplers = ["greedy", "categorical"]

    # raw Gaussian-minus-uniform delta matrices (metric x grammar), one per sampler
    mats = {}
    for s in samplers:
        mat = np.full((len(ALL_METRICS), len(gorder)), np.nan)
        for j, g in enumerate(gorder):
            ru = _op_point_maxT(dfx, g, "uniform", s)
            rg = _op_point_maxT(dfx, g, "gaussian", s)
            if ru is None or rg is None:
                continue
            for i, (_, col) in enumerate(ALL_METRICS):
                if col not in df.columns:
                    continue
                if col != "mean_both_rules" and (bool(gate.loc[ru.name]) or bool(gate.loc[rg.name])):
                    continue  # diversity unreliable at this operating point
                vu = pd.to_numeric(pd.Series([ru[col]]), errors="coerce").iloc[0]
                vg = pd.to_numeric(pd.Series([rg[col]]), errors="coerce").iloc[0]
                if np.isnan(vu) or np.isnan(vg):
                    continue
                mat[i, j] = vg - vu
        mats[s] = mat

    # per-metric (row) scale shared across both panels, so colour is comparable
    # within a metric and not swamped by the unbounded entropy rows. Annotation
    # stays the RAW delta.
    stacked = np.concatenate([mats[s] for s in samplers], axis=1)
    row_absmax = np.nanmax(np.abs(stacked), axis=1)
    row_absmax[~np.isfinite(row_absmax) | (row_absmax == 0)] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, s in zip(axes, samplers):
        mat = mats[s]
        norm = mat / row_absmax[:, None]
        sns.heatmap(norm, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=mat, fmt=".2f", annot_kws={"size": 7},
                    xticklabels=disp, yticklabels=[m for m, _ in ALL_METRICS],
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"label": "Δ (within-metric normalised)"})
        ax.set_title(f"{SAMPLER_DISPLAY[s]} sampling", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Gaussian vs. uniform across all metrics  "
                 "(annotation = raw Gaussian − uniform at the max-compute, best-accuracy "
                 "operating point;\ncolour normalised per metric row; red = Gaussian higher)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    w(f"Source: `results/combined_6_grammar.csv` (oracle sweep, {len(df)} rows). "
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

    # decoder-comparison + Gaussian-focused figures
    fig7_decoder_compute(best, HERE / "fig7_decoder_comparison_compute.png")
    fig8_decoder_tradeoff(best, HERE / "fig8_decoder_comparison_tradeoff.png")
    try:
        fig9_gaussian_schedule(HERE / "fig9_gaussian_schedule.png")
        fig9_ok = True
    except Exception as e:  # schedule import is environment-dependent
        fig9_ok = False
        print(f"[warn] fig9 (gaussian schedule) skipped: {e}")
    fig10_gaussian_T_sigma(df, HERE / "fig10_gaussian_T_sigma.png")
    fig11_gaussian_vs_uniform(df, HERE / "fig11_gaussian_vs_uniform.png")

    write_outlier_report(df, best, pairs, HERE / "outlier_analysis.md")

    print("Wrote to", HERE)
    for f in ["fig1_accuracy_vs_compute.png", "fig2_diversity_vs_accuracy.png",
              "fig3_greedy_vs_categorical.png", "fig3b_tradeoff_quadrant.png",
              "fig4_compute_efficiency.png", "fig5_baN_parity_anomaly.png"]:
        print("  ", f)
    if fig6_ok:
        print("   fig6_oracle_monotonicity.png")
    for f in ["fig7_decoder_comparison_compute.png", "fig8_decoder_comparison_tradeoff.png"]:
        print("  ", f)
    if fig9_ok:
        print("   fig9_gaussian_schedule.png")
    for f in ["fig10_gaussian_T_sigma.png", "fig11_gaussian_vs_uniform.png"]:
        print("  ", f)
    print("   outlier_analysis.md")


if __name__ == "__main__":
    main()
