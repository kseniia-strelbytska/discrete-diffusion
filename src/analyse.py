from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

GRAMMARS = ["aNbN", "aNbNcN", "baN", "bbaN", "parentheses_and_brackets", "not_nested_parentheses_and_brackets"]
STRATEGIES = ["ar", "ebsampler", "gaussian", "uniform"]
SAMPLING_STRATEGIES = ["categorical", "greedy"]

def main():
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
    
    for grammar in GRAMMARS:
        for strategy in STRATEGIES:
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