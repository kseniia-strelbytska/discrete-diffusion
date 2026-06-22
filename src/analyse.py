from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from oracle.grammar_oracles import oracleModel
from datasets.re_grammar import REGrammar
from datasets.constants import *
import torch
from tqdm import tqdm
import random

GRAMMARS = ["baN", "bbaN", "aNbN", "parentheses_and_brackets", "aNbNcN", "not_nested_parentheses_and_brackets"]
# GRAMMARS = ['parentheses_and_brackets', 'not_nested_parentheses_and_brackets']
STRATEGIES = ["ebsampler", "gaussian", "uniform"]
SAMPLING_STRATEGIES = ["categorical", "greedy"]
L = 32

GRAMMAR_MAPPING = {
    "baN": "L1",
    "bbaN": "L2",
    "aNbN": "L3",
    "parentheses_and_brackets": "L4",
    "aNbNcN": "L5",
    "not_nested_parentheses_and_brackets": "L6"
}

def monotonicity_plot(name='monotonicity_dyck_grammars.png'):
    random.seed(42)
    # make a line plot with 4 subplots (2x2):
    xs = [[] for _ in range(len(GRAMMARS))]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    axes = axes.flatten()
    
    for token_idx in range(2):
        axes[token_idx].set_title(f'Logits for token index {token_idx}')
        axes[token_idx].set_xlabel("Position in sequence")
        axes[token_idx].set_ylabel("Logit for 'a' or '['")
        for grammar in tqdm(range(len(GRAMMARS))):
            oracle = oracleModel(GRAMMARS[grammar], vocab_size=8, device='cpu')
            masked_input = torch.tensor([SOS_token] + [MASK_token] * (L + 1), dtype=torch.long).unsqueeze(0)
            logits = oracle(masked_input)[0]
            
            xs[grammar] = logits[:, token_idx][1:]  # placeholder for actual data
            print(xs[grammar])
            axes[token_idx].plot(xs[grammar], label=GRAMMARS[grammar])
    
    # for token_idx in range(2):
    #     axes[2 + token_idx].set_title(f'Logits for token index {token_idx}')
    #     axes[2 + token_idx].set_xlabel("Position in sequence")
    #     axes[2 + token_idx].set_ylabel("Logit for 'a' or '['")
    #     for grammar in tqdm(range(len(GRAMMARS))):
    #         oracle = oracleModel(GRAMMARS[grammar], vocab_size=8, device='cpu')
            
    #         grammarClass = REGrammar(GRAMMARS[grammar], l=L+2)
    #         grammarClass.generate_seq()
            
    #         plausible_input = grammarClass.data[random.randint(0, len(grammarClass.data) - 1)]
    #         plausible_input = torch.where(torch.rand_like(plausible_input, dtype=torch.float64) < 0.2, plausible_input, torch.tensor(MASK_token))
    #         plausible_input[0] = SOS_token
    #         print(plausible_input)
            
    #         logits = oracle(plausible_input.unsqueeze(0))[0]
            
    #         xs[grammar] = logits[:, token_idx][1:]  # placeholder for actual data
    #         axes[2 + token_idx].plot(xs[grammar], label=GRAMMARS[grammar])
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.savefig(name)

def plot_accuracy_vs_compute(df):
    for grammar in GRAMMARS:
        for decoder in STRATEGIES:
            for sampler in SAMPLING_STRATEGIES:
                mode = 'role of gamma' if decoder == 'ebsampler' else 'role of compute'
                id = f"{mode}_{grammar}_{decoder}_{sampler}"
                print(id)
                
                selected = df[(df.grammar == grammar) & (df.strategy == decoder) & (df.sampling_strategy == sampler)]
                if decoder == 'ebsampler':
                    selected = selected.drop_duplicates(subset=['eb_gamma'])
                    selected = selected.sort_values(by='n_steps_mean')
                    plt.plot(selected.n_steps_mean, selected.mean_both_rules, marker='o')
                    for _, row in selected.iterrows():
                        plt.annotate(f"γ={row.eb_gamma}", (row.n_steps_mean, row.mean_both_rules),
                                     textcoords='offset points', xytext=(4, 4), fontsize=7)
                else:
                    selected = selected.sort_values(by='n_steps_mean').reset_index(drop=True)
                    plt.plot(selected.n_steps_mean, selected.mean_both_rules)
                    acc = selected['mean_both_rules'].values
                    for i in range(1, len(acc) - 1):
                        is_peak = acc[i] > acc[i-1] and acc[i] >= acc[i+1]
                        is_trough = acc[i] < acc[i-1] and acc[i] <= acc[i+1]
                        if is_peak or is_trough:
                            row = selected.iloc[i]
                            label = f"(T={int(row['T'])}, σ={row.sigma})" if decoder == 'gaussian' else f"T={int(row['T'])}"
                            plt.annotate(label, (row.n_steps_mean, row.mean_both_rules),
                                         textcoords='offset points', xytext=(4, 4), fontsize=7)
                max_acc = selected.mean_both_rules.max()
                plt.axhline(y=max_acc, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                plt.text(260, max_acc, f'{max_acc:.3f}', va='bottom', ha='right', fontsize=7, color='gray')
                plt.xlabel('n_steps_mean')
                plt.xlim(0, 260)
                plt.ylim(0, 1.05)
                plt.ylabel('mean_both_rules')
                plt.title(f'Accuracy vs Compute for {grammar} with {decoder} and {sampler}')
                plt.savefig(f"./x_figures/{id}.png")
                plt.clf()  # Clear the figure for the next plot
    
def plot_accuracy_vs_compute_uniform(df):
    decoder = 'uniform'
    for sampler in SAMPLING_STRATEGIES:
        plt.figure(figsize=(10, 6))
        for grammar in GRAMMARS:
            selected = df[(df.grammar == grammar) & (df.strategy == decoder) & (df.sampling_strategy == sampler)]
            selected = selected.sort_values(by='n_steps_mean')
            
            idx_of_1_step = selected[selected.n_steps_mean == 1].index
            if not idx_of_1_step.empty:
                print(selected.loc[idx_of_1_step, ['grammar', 'n_steps_mean', 'sampling_strategy','mean_both_rules']])
            
            plt.plot(selected.n_steps_mean, selected.mean_both_rules, label=GRAMMAR_MAPPING[grammar])

        id = f"clean_figure_role_of_compute_{decoder}_{sampler}"
        plt.xlabel('compute (denoising steps made)')
        plt.xlim(0, 260)
        plt.ylabel('Accuracy (satisfying both rules)')
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.title(f'Accuracy vs Compute for {decoder} ({sampler} sampling)')
        plt.savefig(f"./x_clean_figures/{id}.png")
        plt.clf()  # Clear the figure for the next plot

def plot_categorical_and_greedy(df):
    plt.subplots(2, 3, figsize=(15, 10))
    grammar_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for grammar in GRAMMARS:
        pos = grammar_positions[GRAMMARS.index(grammar)]
        ax = plt.subplot2grid((2, 3), pos)
        
        decoder = 'uniform'
        for sampler in SAMPLING_STRATEGIES:
            selected = df[(df.grammar == grammar) & (df.strategy == decoder) & (df.sampling_strategy == sampler)]
            selected = selected.sort_values(by='n_steps_mean')
            ax.plot(selected.n_steps_mean, selected.mean_both_rules, label=f"{sampler}")
    
        id = f"clean_figure_greedy_and_categorical_{decoder}"
        ax.set_title(f"{GRAMMAR_MAPPING[grammar]}")
        ax.set_xlabel('n_steps_mean')
        ax.set_ylabel('mean_both_rules')
        ax.set_xlim(0, 260)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f'Accuracy vs Compute under {decoder} strategy', fontsize=12)
    plt.savefig(f"./x_clean_figures/{id}.png")

# Map each language to its corresponding metrics
DIVERSITY_METRICS = {
    "baN": [
        "uniqueness"
    ],
    "bbaN": [
        "nm_joint_coverage"
    ],
    "aNbN": [
        "n_coverage"
    ],
    "aNbNcN": [
        "n_entropy",
        "n_coverage"
    ],
    "parentheses_and_brackets": [
        "uniqueness"
    ],
    "not_nested_parentheses_and_brackets": [
        "uniqueness"
    ]
}

NEEDED_SETTINGS = {
    "ar": [
        "sampling_strategy",
        "n_steps_mean"
    ],
    "ebsampler": [
        "sampling_strategy",
        "eb_gamma",
        "n_steps_mean"
    ],
    "gaussian": [
        "sampling_strategy",
        "sigma",
        "n_steps_mean"
    ],
    "uniform": [
        "sampling_strategy",
        "n_steps_mean"
    ]       
}

def main():
    # monotonicity_plot(name='monotonicity_dyck_grammars.png')
    # exit(0)
    
    # get file as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()
    # folder in parent directory, resolve path
    resolved_path = Path(args.file).parent.resolve()
    resolved_path = Path(resolved_path) / Path(args.file).name
    
    print(f"Opening file: {resolved_path}")
    
    # read csv
    df = pd.read_csv(resolved_path)
    
    # plot_accuracy_vs_compute(df)
    # plot_accuracy_vs_compute_uniform(df)
    plot_categorical_and_greedy(df)
    
    exit(0)
    
    # Q1: Does AR actually hit 1.0 on every grammar? CONFRIMED
    working_slice = df[df.strategy=='ar'][['grammar', 'sampling_strategy', 'mean_both_rules', 'n_steps_mean']]
    
    # Q2: What's the minimum compute each (strategy, sampler) needs to reach 0.95 accuracy on each grammar?
    """ 
    The answer is a 4-grammar × 4-decoder x 2-sampler table. 
    For each (grammar, strategy, sampler), filter rows with mean_both_rules >= 0.95, take the minimum n_steps_mean. 
    If no rows clear 0.95, write "—"
    """
    
    # save the results with all possible diversity metrics as columns in a new dataframe
    results = pd.DataFrame(columns=['grammar', 'strategy', 'sampling_strategy'] + ['mean_both_rules', 'n_steps_mean'] + ['n_entropy', 'm_entropy', 'nm_joint_coverage', 'n_coverage', 'dfa_state_coverage', 'dfa_transition_coverage', 'max_depth_ratio_mean', 'max_depth_ratio_std', 'brackets_parens_ratio_mean', 'brackets_parens_ratio_std', 'n_zero_paren_sequences'])
    
    diversity = False
    
    # for grammar in GRAMMARS:
    #     for strategy in STRATEGIES:
    #         for sampler in SAMPLING_STRATEGIES:
    #             filtered = df[(df.grammar==grammar) & (df.strategy==strategy) & (df.sampling_strategy==sampler)]
    #             filtered = filtered[filtered.mean_both_rules >= 0.95]
    #             # format each print in rigid columns
    #             if len(filtered) == 0:
    #                 print(f"{grammar:<40} {strategy:<20} {sampler:>20}: —")
    #             else:
    #                 # find the line with the smallest n_steps_mean and print it
    #                 line = filtered.loc[filtered.n_steps_mean.idxmin()]
    #                 if diversity:
    #                     important_traits = DIVERSITY_METRICS[grammar]
    #                     line_important_traits = line[important_traits + ['mean_both_rules', 'n_steps_mean']]
                        
    #                     min_steps = filtered.n_steps_mean.min()
    #                     print(f"{grammar:<40} {strategy:<20} {sampler:>20}: {min_steps:>5.2f} steps, mean_both_rules: {line.mean_both_rules:.4f}, important traits: {line_important_traits.to_dict()}")
    #                 else:
    #                     min_steps = filtered.n_steps_mean.min()
    #                     print(f"{grammar:<40} {strategy:<20} {sampler:>20}: {min_steps:>5.2f} steps, mean_both_rules: {line.mean_both_rules:.4f}")

    with open('results_summary.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for grammar in GRAMMARS:
            for strategy in STRATEGIES:
                for sampler in SAMPLING_STRATEGIES:
                    best_rows = df[(df.grammar==grammar) & (df.strategy==strategy) & (df.sampling_strategy == sampler)].sort_values(by='mean_both_rules', ascending=False).head(5)
                    
                    for idx, best_row in best_rows.iterrows():
                        results = pd.concat([results, pd.DataFrame([best_row])], ignore_index=True)
                        
                        settings = NEEDED_SETTINGS[strategy]
                        settings += DIVERSITY_METRICS[grammar]
                        settings_str = ', '.join([f"{setting}={best_row[setting]}" for setting in settings])
                        
                        print(f"{grammar:<40} {strategy:<20}: best settings: {settings_str}, mean_both_rules: {best_row.mean_both_rules:.4f}")
                
            print('' + '-'*80 + '\n')
    
if __name__ == "__main__":
    main()