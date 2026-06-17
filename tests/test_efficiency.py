"""
Oracle grammar efficiency benchmark.

Usage:
    python tests/test_efficiency.py [--seq-length 8] [--n-samples 100]
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import torch

import datasets.re_data as rd
from datasets.constants import PAD_token
from oracle.grammar_oracles import (
    aNbN_get_marginals,
    baN_get_marginals,
    bbaN_get_marginals,
    aNbNcN_get_marginals,
    parentheses_and_brackets_get_marginals,
    not_nested_parentheses_and_brackets_get_marginals,
)

ORACLES = [
    ('aNbN',                                aNbN_get_marginals,                                6),
    ('baN',                                 baN_get_marginals,                                 6),
    ('bbaN',                                bbaN_get_marginals,                                6),
    ('aNbNcN',                              aNbNcN_get_marginals,                              7),
    ('parentheses_and_brackets',            parentheses_and_brackets_get_marginals,            8),
    ('not_nested_parentheses_and_brackets', not_nested_parentheses_and_brackets_get_marginals, 8),
]


def _pad_to(arr, length):
    arr = np.asarray(arr, dtype=int)
    pad = length - len(arr)
    if pad > 0:
        arr = np.concatenate([arr, np.full(pad, PAD_token)])
    return arr[:length]


def generate_seqs(grammar_name, seq_length, n_samples):
    """
    Generate n_samples random valid sequences padded to seq_length.

    Passes (seq_length - 2) as max_length to re_data generators so the raw
    sequence (SOS + content + EOS) is at most seq_length tokens, then pads
    shorter sequences with PAD.
    """
    ml = seq_length - 2  # content budget: keeps total <= seq_length
    if grammar_name == 'aNbN':
        raw = rd.generate_aNbN_grammar_data(n_samples, ml, all_sequences=False)
    elif grammar_name == 'baN':
        raw = rd.generate_baN_grammar_data(n_samples, ml)
    elif grammar_name == 'bbaN':
        raw = rd.generate_bbaN_grammar_data(n_samples, ml)
    elif grammar_name == 'aNbNcN':
        raw = rd.generate_aNbNcN_grammar_data(n_samples, ml, all_sequences=False)
    elif grammar_name == 'parentheses_and_brackets':
        raw = rd.generate_matched_parentheses_and_brackets_data(n_samples, ml)
    elif grammar_name == 'not_nested_parentheses_and_brackets':
        raw = rd.generate_not_nested_matched_parentheses_and_brackets_data(n_samples, ml)
    else:
        raise ValueError(f"Unknown grammar: {grammar_name}")

    return [torch.tensor(_pad_to(a, seq_length), dtype=torch.long) for a in raw]


def benchmark(seq_length: int, n_samples: int = 100):
    header = f"Oracle grammar benchmark  |  seq_length={seq_length}  n_samples={n_samples}"
    sep = "─" * len(header)
    print(f"\n{header}\n{sep}")
    print(f"  {'grammar':<44} {'avg µs':>8}  {'min µs':>8}  {'max µs':>8}")
    print(f"  {'':─<44} {'':─>8}  {'':─>8}  {'':─>8}")

    for name, oracle_fn, vocab_size in ORACLES:
        seqs = generate_seqs(name, seq_length, n_samples)
        times = []
        for seq in seqs:
            t0 = time.perf_counter()
            oracle_fn(seq, vocab_size)
            times.append((time.perf_counter() - t0) * 1e6)
        avg, lo, hi = sum(times) / len(times), min(times), max(times)
        print(f"  {name:<44} {avg:>8.1f}  {lo:>8.1f}  {hi:>8.1f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Oracle grammar efficiency benchmark")
    parser.add_argument("--seq-length", type=int, default=8,
                        help="Sequence length including SOS and EOS (default: 8)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of random inputs to average over (default: 100)")
    args = parser.parse_args()
    benchmark(args.seq_length, args.n_samples)


if __name__ == "__main__":
    main()
