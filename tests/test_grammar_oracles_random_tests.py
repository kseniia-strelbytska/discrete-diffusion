"""
Random-sampling oracle tests.

For each grammar we:
  1. Draw a pool of valid sequences via datasets.re_data generators.
  2. For each of n_samples random (seq, masking_pattern) pairs, verify:
       a. Oracle returns non-None  (every masked valid sequence has a valid completion)
       b. Probabilities lie in [0, 1] and sum to 1 at every position
       c. Every unmasked position receives probability 1 for its actual token
  3. For grammars whose valid sequence set is fully enumerable (aNbN, aNbNcN),
     the pool contains ALL valid sequences up to seq_length, so we additionally
     compare oracle marginals against exact brute-force marginals over the pool.

Usage:
    pytest tests/test_grammar_oracles_random_tests.py --n-samples 100000
    pytest tests/test_grammar_oracles_random_tests.py --n-samples 100000 --seq-length 20

Token scheme: A=0, B=1, EOS=2, SOS=3, PAD=4, MASK=5, C=6
              open_paren=0, close_paren=1, open_bracket=6, close_bracket=7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import random as py_random
import time

import numpy as np
import pytest
import torch

from datasets.constants import SOS_token, EOS_token, PAD_token, MASK_token
from datasets.re_data import (
    generate_aNbN_grammar_data,
    generate_baN_grammar_data,
    generate_bbaN_grammar_data,
    generate_aNbNcN_grammar_data,
    generate_matched_parentheses_and_brackets_data,
    generate_not_nested_matched_parentheses_and_brackets_data,
    pad,
)
from oracle.grammar_oracles import (
    aNbN_get_marginals,
    baN_get_marginals,
    bbaN_get_marginals,
    aNbNcN_get_marginals,
    parentheses_and_brackets_get_marginals,
    not_nested_parentheses_and_brackets_get_marginals,
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_pool(raw_seqs: list, target_length: int):
    """
    Pad raw_seqs (list of numpy arrays of varying length) to target_length
    and return (list[Tensor[L]], Tensor[N, L]).
    Sequences longer than target_length are dropped.
    """
    filtered = [s for s in raw_seqs if len(s) <= target_length]
    if not filtered:
        raise ValueError(
            f"All {len(raw_seqs)} generated sequences exceed target_length={target_length}. "
            "Increase --seq-length."
        )
    padded = pad(filtered, max_seq_length=target_length)
    seqs = [torch.tensor(row, dtype=torch.long) for row in padded]
    return seqs, torch.stack(seqs)


def _random_mask(seq: torch.Tensor, rng: py_random.Random) -> torch.Tensor:
    """Randomly mask a subset of positions 1..L-1; SOS at pos 0 is never masked."""
    masked = seq.clone()
    for pos in range(1, seq.shape[0]):
        if rng.random() < 0.5:
            masked[pos] = MASK_token
    return masked


def _check_properties(oracle_fn, vocab_size, seq, masked_seq, atol=1e-4):
    """
    Run oracle_fn on masked_seq and verify correctness properties.
    Returns an error string on failure, None on success.
    """
    status, result = oracle_fn(masked_seq, vocab_size)

    # (a) Valid masked sequence must always have a valid completion.
    if status is None:
        return (
            f"Oracle returned None for masked={masked_seq.tolist()} "
            f"(original valid seq={seq.tolist()})"
        )

    # (b) Probabilities in [0, 1].
    if (result < -atol).any() or (result > 1.0 + atol).any():
        return (
            f"Probs out of [0, 1]: min={result.min():.6f}, max={result.max():.6f} "
            f"seq={masked_seq.tolist()}"
        )

    # (b) Probabilities sum to 1 at every position.
    row_sums = result.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol):
        bad = [(i, float(v)) for i, v in enumerate(row_sums) if abs(float(v) - 1.0) > atol]
        return f"Row sums != 1 at positions {bad}  seq={masked_seq.tolist()}"

    # (c) Every unmasked position is deterministic.
    for pos in range(masked_seq.shape[0]):
        if masked_seq[pos].item() != MASK_token:
            tok = seq[pos].item()
            p = result[pos, tok].item()
            if abs(p - 1.0) > atol:
                return (
                    f"Pos {pos} not deterministic: token={tok}, prob={p:.6f} (expected 1.0) "
                    f"seq={masked_seq.tolist()}"
                )
    return None


def _brute_force_marginals(masked_seq: torch.Tensor, pool: torch.Tensor, vocab_size: int):
    """
    Compute marginals over all pool sequences consistent with masked_seq.
    pool: (N, L) tensor.  Returns Tensor[L, vocab_size] or None.
    """
    consistent_mask = (
        (pool == masked_seq.unsqueeze(0)) | (masked_seq.unsqueeze(0) == MASK_token)
    ).all(dim=1)
    consistent = pool[consistent_mask]
    if consistent.shape[0] == 0:
        return None
    L = masked_seq.shape[0]
    marginals = torch.zeros(L, vocab_size, dtype=torch.float32)
    for tok_id in range(vocab_size):
        marginals[:, tok_id] = (consistent == tok_id).float().mean(dim=0)
    return marginals


def _run_test(
    oracle_fn, vocab_size, pool_seqs, pool_tensor, n_samples, rng,
    label='', atol=1e-4, exact_brute_force=False,
):
    """
    Sample n_samples random (seq, mask) pairs from pool_seqs, verify oracle
    properties, and optionally compare against exact brute-force marginals.
    """
    failures = []
    total = 0
    oracle_total_s = 0.0
    n_pool = len(pool_seqs)

    for _ in range(n_samples):
        seq = pool_seqs[rng.randint(0, n_pool - 1)]
        masked_seq = _random_mask(seq, rng)
        total += 1

        t0 = time.perf_counter()
        err = _check_properties(oracle_fn, vocab_size, seq, masked_seq, atol=atol)
        oracle_total_s += time.perf_counter() - t0

        if err:
            failures.append(err)
            if len(failures) >= 10:
                break
            continue

        if exact_brute_force:
            _, result = oracle_fn(masked_seq, vocab_size)
            bf = _brute_force_marginals(masked_seq, pool_tensor, vocab_size)
            if bf is not None and not torch.allclose(result, bf, atol=atol):
                max_diff = (result - bf).abs().max().item()
                failures.append(
                    f"Oracle ≠ brute-force (max_diff={max_diff:.6f}) "
                    f"seq={masked_seq.tolist()}"
                )
                if len(failures) >= 10:
                    break

    if total:
        avg_us = oracle_total_s / total * 1e6
        print(f"\n[{label}] oracle avg: {avg_us:.1f} µs/call over {total} samples")

    if failures:
        msg = f"{label}: {len(failures)} failures\n"
        for f in failures[:5]:
            msg += f"  {f}\n"
        pytest.fail(msg)


# ─── test classes ─────────────────────────────────────────────────────────────

class TestANBNRandom:
    """
    aNbN: all valid sequences up to seq_length are enumerated (all_sequences=True),
    so brute-force comparison is exact.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length):
        raw = generate_aNbN_grammar_data(
            num_samples=1, max_length=seq_length - 2, all_sequences=True,
        )
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid aNbN sequences generated"

    def test_oracle_properties_and_brute_force(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            aNbN_get_marginals, vocab_size=6,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='aNbN', exact_brute_force=True,
        )


class TestBaNRandom:
    """
    baN: pool is a random sample of valid sequences; property checks only.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length, n_samples):
        pool_size = max(n_samples // 10, 1000)
        raw = generate_baN_grammar_data(num_samples=pool_size, max_length=seq_length - 2)
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid baN sequences generated"

    def test_oracle_properties(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            baN_get_marginals, vocab_size=6,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='baN', exact_brute_force=False,
        )

    def test_invalid_returns_none(self, seq_length):
        # position 1 = A → invalid (baN requires B at pos 1)
        seq = torch.tensor(
            [SOS_token, 0, 1, EOS_token] + [PAD_token] * (seq_length - 4),
            dtype=torch.long,
        )
        status, _ = baN_get_marginals(seq, vocab_size=6)
        assert status is None

    def test_position1_masked_gives_B(self, seq_length):
        # SOS MASK B EOS PAD... — pos 1 masked, rest valid → oracle must put prob 1 on B
        B = 1
        seq = torch.tensor(
            [SOS_token, MASK_token, B, EOS_token] + [PAD_token] * (seq_length - 4),
            dtype=torch.long,
        )
        status, result = baN_get_marginals(seq, vocab_size=6)
        if status is not None:
            assert abs(result[1, B].item() - 1.0) < 1e-4, \
                f"pos 1 prob(B)={result[1, B].item():.4f}, expected 1.0"


class TestBBaNRandom:
    """
    bbaN: pool is a random sample of valid sequences; property checks only.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length, n_samples):
        pool_size = max(n_samples // 10, 1000)
        raw = generate_bbaN_grammar_data(num_samples=pool_size, max_length=seq_length - 2)
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid bbaN sequences generated"

    def test_oracle_properties(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            bbaN_get_marginals, vocab_size=6,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='bbaN', exact_brute_force=False,
        )

    def test_A_before_B_returns_none(self, seq_length):
        A, B = 0, 1
        seq = torch.tensor(
            [SOS_token, A, B, EOS_token] + [PAD_token] * (seq_length - 4),
            dtype=torch.long,
        )
        status, _ = bbaN_get_marginals(seq, vocab_size=6)
        assert status is None


class TestANBNCNRandom:
    """
    aNbNcN: all valid sequences up to seq_length are enumerated (all_sequences=True),
    so brute-force comparison is exact.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length):
        raw = generate_aNbNcN_grammar_data(
            num_samples=1, max_length=seq_length - 2, all_sequences=True,
        )
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid aNbNcN sequences generated"

    def test_oracle_properties_and_brute_force(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            aNbNcN_get_marginals, vocab_size=7,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='aNbNcN', exact_brute_force=True,
        )

    def test_wrong_order_returns_none(self, seq_length):
        A, B, C = 0, 1, 6
        seq = torch.tensor(
            [SOS_token, B, A, C, EOS_token] + [PAD_token] * (seq_length - 5),
            dtype=torch.long,
        )
        status, _ = aNbNcN_get_marginals(seq, vocab_size=7)
        assert status is None


class TestParenthesesAndBracketsRandom:
    """
    Nested Dyck: pool is a random sample; property checks only.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length, n_samples):
        pool_size = max(n_samples // 10, 1000)
        raw = generate_matched_parentheses_and_brackets_data(
            num_samples=pool_size, max_length=seq_length - 2,
        )
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid parentheses_and_brackets sequences generated"

    def test_oracle_properties(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            parentheses_and_brackets_get_marginals, vocab_size=8,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='parentheses_and_brackets', exact_brute_force=False,
        )

    def test_wrong_close_returns_none(self):
        OPEN_P, CLOSE_B = 0, 7
        seq = torch.tensor(
            [SOS_token, OPEN_P, CLOSE_B, EOS_token, PAD_token, PAD_token, PAD_token, PAD_token],
            dtype=torch.long,
        )
        status, _ = parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        assert status is None


class TestNotNestedParenthesesAndBracketsRandom:
    """
    Not-nested (independent) Dyck: pool is a random sample; property checks only.
    """

    @pytest.fixture(scope='class')
    def pool(self, seq_length, n_samples):
        pool_size = max(n_samples // 10, 1000)
        raw = generate_not_nested_matched_parentheses_and_brackets_data(
            num_samples=pool_size, max_length=seq_length - 2,
        )
        return _make_pool(raw, target_length=seq_length)

    def test_pool_nonempty(self, pool):
        seqs, _ = pool
        assert len(seqs) > 0, "No valid not_nested sequences generated"

    def test_oracle_properties(self, pool, n_samples):
        seqs, pool_tensor = pool
        _run_test(
            not_nested_parentheses_and_brackets_get_marginals, vocab_size=8,
            pool_seqs=seqs, pool_tensor=pool_tensor,
            n_samples=n_samples, rng=py_random.Random(42),
            label='not_nested_paren_bracket', exact_brute_force=False,
        )

    def test_close_before_open_returns_none(self):
        CLOSE_P, OPEN_P = 1, 0
        seq = torch.tensor(
            [SOS_token, CLOSE_P, OPEN_P, EOS_token, PAD_token, PAD_token, PAD_token, PAD_token],
            dtype=torch.long,
        )
        status, _ = not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        assert status is None

    def test_nested_rejected_by_not_nested(self):
        # ( ] ) [ — close_b arrives before any open_b → invalid even for not-nested
        OPEN_P, CLOSE_B, CLOSE_P, OPEN_B = 0, 7, 1, 6
        seq = torch.tensor(
            [SOS_token, OPEN_P, CLOSE_B, CLOSE_P, OPEN_B, EOS_token, PAD_token, PAD_token],
            dtype=torch.long,
        )
        status, _ = not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        assert status is None
