"""
plot_T_sweep.py
===============
Read the CSV produced by eval_oracle_T_sweep.py and emit two kinds of figures:

  (1) Kitchen-sink dashboards: one PNG per grammar, panels for every metric.
      Per-panel legend identifies each curve including its hyperparameter.
  (2) Paper Pareto figure: a multi-panel grid with accuracy-vs-compute curves.

Usage
-----
  python src/oracle/plot_T_sweep.py --csv results/T-sweep-nonDyck/oracle_T_sweep_results.csv \
      --out-dir results/T-sweep-nonDyck/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants matching eval_oracle_T_sweep.py
# ---------------------------------------------------------------------------
STATS_NAMES  = ['rule1', 'rule2', 'both_rules', 'format']
PRIMARY_STAT = 'both_rules'

UNIVERSAL_DIV = [
    'uniqueness', 'duplication_rate',
    'mean_lev_dist_normalized',
    'bigram_diversity', 'trigram_diversity',
]
GRAMMAR_SPECIFIC_DIV = {
    'baN':                                 ['dfa_state_coverage', 'dfa_transition_coverage'],
    'bbaN':                                ['dfa_state_coverage', 'dfa_transition_coverage',
                                            'n_entropy', 'm_entropy', 'nm_joint_coverage'],
    'aNbN':                                ['n_entropy', 'n_coverage'],
    'aNbNcN':                              ['n_entropy', 'n_coverage'],
    'parentheses_and_brackets':            ['max_depth_ratio_mean', 'max_depth_ratio_std',
                                            'brackets_parens_ratio_mean', 'brackets_parens_ratio_std',
                                            'n_zero_paren_sequences'],
    'not_nested_parentheses_and_brackets': ['max_depth_ratio_mean', 'max_depth_ratio_std',
                                            'brackets_parens_ratio_mean', 'brackets_parens_ratio_std',
                                            'n_zero_paren_sequences'],
}

STRATEGY_COLOURS = {
    'ar':        '#000000',
    'uniform':   '#1f77b4',
    'gaussian':  '#2ca02c',
    'ebsampler': '#d62728',
}
SAMPLER_LINESTYLE = {'greedy': '-', 'categorical': '--'}


def load_csv(path):
    df = pd.read_csv(path)
    num_cols = (
        ['T', 'L', 'n_reps', 'n_steps_mean', 'n_steps_max', 'elapsed_s',
         'sigma', 'eb_gamma']
        + [f'mean_{n}' for n in STATS_NAMES]
        + [f'std_{n}'  for n in STATS_NAMES]
        + [c for c in UNIVERSAL_DIV]
        + [c for vals in GRAMMAR_SPECIFIC_DIV.values() for c in vals]
        + ['n_correct', 'lev_n_used']
    )
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['deterministic'] = df['deterministic'].astype(str).str.lower() == 'true'
    return df


def hyperparam_value(row):
    s = row['strategy']
    if s == 'gaussian':
        return row.get('sigma')
    if s == 'ebsampler':
        return row.get('eb_gamma')
    return None


# ---------------------------------------------------------------------------
# Kitchen-sink dashboard
# ---------------------------------------------------------------------------

def dashboard_panels_for_grammar(grammar):
    panels = []
    for stat in ['rule1', 'rule2', 'both_rules', 'format']:
        panels.append(('accuracy', f'mean_{stat}', stat, (0.0, 1.05)))
    panels.append(('steps', 'n_steps_mean', 'n_steps_mean', None))
    panels.append(('steps', 'n_steps_max',  'n_steps_max',  None))
    for m in UNIVERSAL_DIV:
        panels.append(('diversity', m, m, None))
    for m in GRAMMAR_SPECIFIC_DIV.get(grammar, []):
        panels.append(('diversity', m, m, None))
    return panels


def _curve_label(strategy, hyperparam, sampler):
    samp = 'gr' if sampler == 'greedy' else 'cat'
    if strategy == 'ar':
        return f'AR·{samp}'
    if strategy == 'uniform':
        return f'unif·{samp}'
    if strategy == 'gaussian':
        return f'gauss σ={hyperparam:g}·{samp}'
    if strategy == 'ebsampler':
        return f'EB γ={hyperparam:g}·{samp}'
    return f'{strategy}·{samp}'


def _curve_colour(strategy, hyperparam, hyperparam_list):
    base = STRATEGY_COLOURS[strategy]
    if hyperparam is None or hyperparam_list is None or len(hyperparam_list) < 2:
        return base
    sorted_hp = sorted(hyperparam_list)
    pos = sorted_hp.index(hyperparam) / max(1, len(sorted_hp) - 1)
    from matplotlib.colors import to_rgb
    base_rgb = np.array(to_rgb(base))
    light_rgb = 1.0 - 0.7 * (1.0 - base_rgb)
    out = light_rgb + pos * (base_rgb - light_rgb)
    return tuple(out)


def plot_dashboard(df, grammar, L, out_path):
    sub = df[(df['grammar'] == grammar) & (df['L'] == L)].copy()
    if sub.empty:
        print(f'  no rows for grammar={grammar} L={L}; skipping dashboard')
        return

    panels = dashboard_panels_for_grammar(grammar)
    n_panels = len(panels)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows),
                             squeeze=False)
    fig.suptitle(f'{grammar}  (L={L})  — T-sweep dashboard', fontsize=13, y=0.995)

    curves = []
    for (strategy, sampler), grp_ss in sub.groupby(['strategy', 'sampling_strategy']):
        if strategy in ('uniform', 'ar'):
            curves.append(((strategy, None, sampler), grp_ss))
            continue
        for hp_val, grp in grp_ss.groupby(
                'sigma' if strategy == 'gaussian' else 'eb_gamma'):
            curves.append(((strategy, float(hp_val), sampler), grp))

    sigma_list = sorted(sub.loc[sub['strategy'] == 'gaussian',  'sigma'].dropna().unique().tolist())
    gamma_list = sorted(sub.loc[sub['strategy'] == 'ebsampler', 'eb_gamma'].dropna().unique().tolist())

    def colour_for(strategy, hp):
        if strategy == 'gaussian':
            return _curve_colour(strategy, hp, sigma_list)
        if strategy == 'ebsampler':
            return _curve_colour(strategy, hp, gamma_list)
        return STRATEGY_COLOURS[strategy]

    for idx, (kind, col, title, ylim) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if col not in sub.columns:
            ax.set_title(f'{title}  (column missing)')
            ax.axis('off')
            continue

        for (strategy, hp, sampler), g in curves:
            g_sorted = g.sort_values('n_steps_mean')
            if g_sorted[col].dropna().empty:
                continue
            ls    = SAMPLER_LINESTYLE.get(sampler, '-')
            c     = colour_for(strategy, hp)
            label = _curve_label(strategy, hp, sampler)
            ax.plot(g_sorted['n_steps_mean'], g_sorted[col],
                    marker='o', markersize=3, linewidth=1.2,
                    linestyle=ls, color=c, alpha=0.85, label=label)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('mean n_steps', fontsize=8)
        ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=4.5, loc='best', framealpha=0.7, ncol=2)

    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')


# ---------------------------------------------------------------------------
# Paper Pareto figure
# ---------------------------------------------------------------------------

def _best_per_T(df, strategy, sampler, primary_stat):
    sub = df[(df['strategy'] == strategy) & (df['sampling_strategy'] == sampler)].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(f'mean_{primary_stat}', ascending=False)
    best = sub.drop_duplicates(subset=['T'], keep='first')
    return best.sort_values('T')


def plot_pareto(df, grammars, L, out_path, primary_stat=PRIMARY_STAT):
    n = len(grammars)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    fig.suptitle(f'Accuracy–compute Pareto  ({primary_stat}, L={L})',
                 fontsize=12, y=0.995)

    for i, grammar in enumerate(grammars):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df['grammar'] == grammar) & (df['L'] == L)]
        if sub.empty:
            ax.set_title(grammar + ' (no data)')
            ax.axis('off')
            continue

        for strategy in ['uniform', 'gaussian', 'ebsampler']:
            for sampler in ['greedy', 'categorical']:
                best = _best_per_T(sub, strategy, sampler, primary_stat)
                if best.empty:
                    continue
                ls    = SAMPLER_LINESTYLE[sampler]
                c     = STRATEGY_COLOURS[strategy]
                y     = best[f'mean_{primary_stat}']
                yerr  = best[f'std_{primary_stat}']
                ax.errorbar(best['n_steps_mean'], y, yerr=yerr,
                            fmt='o-', color=c, linestyle=ls,
                            markersize=4, linewidth=1.2, capsize=2,
                            alpha=0.9, label=f'{strategy} · {sampler}')

        ar_rows = sub[sub['strategy'] == 'ar']
        for _, ar_row in ar_rows.iterrows():
            ax.axhline(ar_row[f'mean_{primary_stat}'],
                       color='black', linestyle=':', linewidth=1.0, alpha=0.5)
            ax.plot([ar_row['n_steps_mean']], [ar_row[f'mean_{primary_stat}']],
                    marker='*', markersize=10, color='black',
                    label=f'AR · {ar_row["sampling_strategy"]}')

        ax.set_xlabel('mean n_steps (compute)', fontsize=9)
        ax.set_ylabel(primary_stat, fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(grammar, fontsize=10)
        ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='lower right', ncol=1, framealpha=0.85)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Plot eval_oracle_T_sweep results.')
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--grammars', nargs='+', default=None)
    p.add_argument('--L', type=int, default=None)
    p.add_argument('--primary-stat', type=str, default=PRIMARY_STAT,
                   choices=STATS_NAMES)
    return p.parse_args()


def main():
    args = parse_args()
    df = load_csv(args.csv)
    L = args.L if args.L is not None else int(df['L'].min())
    grammars = args.grammars if args.grammars else list(df.loc[df['L'] == L, 'grammar'].unique())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== Kitchen-sink dashboards ===')
    for g in grammars:
        plot_dashboard(df, g, L, out_dir / f'dashboard_{g}_L{L}.png')

    print('\n=== Paper Pareto figure ===')
    plot_pareto(df, grammars, L,
                out_dir / f'pareto_{args.primary_stat}_L{L}.png',
                primary_stat=args.primary_stat)

    print(f'\nDone. Figures in {out_dir}')


if __name__ == '__main__':
    main()