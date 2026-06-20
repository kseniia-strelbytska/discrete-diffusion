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

GRAMMARS = ["aNbN", "aNbNcN", "baN", "bbaN", "parentheses_and_brackets", "not_nested_parentheses_and_brackets"]
STRATEGIES = ["ar", "ebsampler", "gaussian", "uniform"]
SAMPLING_STRATEGIES = ["categorical", "greedy"]
L = 32

def main():
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
    plt.savefig("monotonicity_empty_grammars.png")
    
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
    
    # Q1: Does AR actually hit 1.0 on every grammar? CONFRIMED
    working_slice = df[df.strategy=='ar'][['grammar', 'sampling_strategy', 'mean_both_rules', 'n_steps_mean']]
    
    # Q2: What's the minimum compute each (strategy, sampler) needs to reach 0.95 accuracy on each grammar?
    """ 
    The answer is a 4-grammar × 4-decoder x 2-sampler table. 
    For each (grammar, strategy, sampler), filter rows with mean_both_rules >= 0.95, take the minimum n_steps_mean. 
    If no rows clear 0.95, write "—"
    """
    for strategy in STRATEGIES:
        for grammar in GRAMMARS:
            for sampler in SAMPLING_STRATEGIES:
                filtered = df[(df.grammar==grammar) & (df.strategy==strategy) & (df.sampling_strategy==sampler)]
                filtered = filtered[filtered.mean_both_rules >= 0.95]
                # format each print in rigid columns
                if len(filtered) == 0:
                    print(f"{grammar:<40} {strategy:<20} {sampler:>20}: —")
                else:
                    min_steps = filtered.n_steps_mean.min()
                    print(f"{grammar:<40} {strategy:<20} {sampler:>20}: {min_steps}")
    
    # display slice
    print("Results:")
    print(working_slice)
    
if __name__ == "__main__":
    main()