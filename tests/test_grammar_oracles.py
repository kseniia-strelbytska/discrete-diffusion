"""
Comprehensive oracle tests: compare each oracle's get_marginals() against
brute-force enumeration on sequences up to length 10.

Token scheme: A=0, B=1, EOS=2, SOS=3, PAD=4, MASK=5, C=6
              open_paren=0, close_paren=1, open_bracket=6, close_bracket=7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import time
import pytest
import torch
from itertools import product, combinations

from datasets.constants import SOS_token, EOS_token, PAD_token, MASK_token
from oracle.grammar_oracles import (
    baN_get_marginals,
    bbaN_get_marginals,
    aNbNcN_get_marginals,
    not_nested_parentheses_and_brackets_get_marginals,
    parentheses_and_brackets_get_marginals,
    aNbN_get_marginals,
)

A = 0; B = 1; C = 6
OPEN_P = 0; CLOSE_P = 1
OPEN_B = 6; CLOSE_B = 7

# ─── brute-force helpers ──────────────────────────────────────────────────────

def is_consistent(partial, complete):
    """True iff complete agrees with partial on every non-MASK position."""
    for p, c in zip(partial.tolist(), complete.tolist()):
        if p != MASK_token and p != c:
            return False
    return True


def brute_force_marginals(partial, valid_seqs, vocab_size):
    """
    Compute marginals by averaging over all valid completions consistent
    with `partial`.  Returns None if no consistent completion exists.
    """
    consistent = [s for s in valid_seqs if is_consistent(partial, s)]
    if not consistent:
        return None
    L = partial.shape[0]
    counts = torch.zeros(L, vocab_size, dtype=torch.float64)
    for s in consistent:
        for p in range(L):
            counts[p, s[p].item()] += 1
    return (counts / len(consistent)).float()


def all_masking_patterns(seq):
    """Yield all 2^{L-1} sequences obtained by masking any subset of pos 1..L-1."""
    L = seq.shape[0]
    maskable = list(range(1, L))
    for bits in product([False, True], repeat=len(maskable)):
        masked = seq.clone()
        for pos, do_mask in zip(maskable, bits):
            if do_mask:
                masked[pos] = MASK_token
        yield masked


# ─── brute-force generators (valid sequences of exactly length L) ─────────────

def gen_aNbN(L):
    seqs = []
    for n in range(1, (L - 2) // 2 + 1):
        pad = L - (2 * n + 2)
        if pad < 0:
            continue
        seq = [SOS_token] + [A]*n + [B]*n + [EOS_token] + [PAD_token]*pad
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def gen_baN(L):
    seqs = []
    for n in range(1, L - 1):
        ep = n + 1
        if ep >= L:
            break
        pad = L - ep - 1
        # all {A,B}^{n-1} where total #A is even (pos1=B contributes 0 A's)
        for bits in product([A, B], repeat=n - 1):
            if sum(1 for x in bits if x == A) % 2 == 0:
                seq = [SOS_token, B] + list(bits) + [EOS_token] + [PAD_token]*pad
                seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def gen_bbaN(L):
    seqs = []
    for n in range(1, L):
        for m in range(0, L // 2 + 1):
            content = n + 2 * m
            ep = content + 1
            if ep >= L:
                break
            pad = L - ep - 1
            seq = [SOS_token] + [B]*n + [A]*(2*m) + [EOS_token] + [PAD_token]*pad
            seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def gen_aNbNcN(L):
    seqs = []
    for n in range(1, (L - 2) // 3 + 1):
        ep = 3 * n + 1
        if ep >= L:
            break
        pad = L - ep - 1
        seq = [SOS_token] + [A]*n + [B]*n + [C]*n + [EOS_token] + [PAD_token]*pad
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def gen_dyck_not_nested(L):
    """
    SOS [Dyck-independent content of length n] EOS PAD*
    Content: interleaved OPEN_P/CLOSE_P and OPEN_B/CLOSE_B, each independently matched.
    Enumerate by DP.
    """
    seqs = []
    tokens = [OPEN_P, CLOSE_P, OPEN_B, CLOSE_B]
    for n in range(2, L - 1, 2):
        ep = n + 1
        if ep >= L:
            break
        pad = L - ep - 1
        # enumerate all length-n strings over tokens where paren-subseq and bracket-subseq are matched
        for content in product(tokens, repeat=n):
            # check paren subsequence
            dp = 0
            ok_p = True
            for t in content:
                if t == OPEN_P: dp += 1
                elif t == CLOSE_P:
                    dp -= 1
                    if dp < 0: ok_p = False; break
            if not ok_p or dp != 0:
                continue
            # check bracket subsequence
            db = 0
            ok_b = True
            for t in content:
                if t == OPEN_B: db += 1
                elif t == CLOSE_B:
                    db -= 1
                    if db < 0: ok_b = False; break
            if not ok_b or db != 0:
                continue
            seq = [SOS_token] + list(content) + [EOS_token] + [PAD_token]*pad
            seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def gen_dyck_nested(L):
    """
    SOS [properly nested () and [] content of length n] EOS PAD*
    """
    seqs = []
    tokens = [OPEN_P, CLOSE_P, OPEN_B, CLOSE_B]
    open_to_close = {OPEN_P: CLOSE_P, OPEN_B: CLOSE_B}
    close_to_open = {CLOSE_P: OPEN_P, CLOSE_B: OPEN_B}

    for n in range(2, L - 1, 2):
        ep = n + 1
        if ep >= L:
            break
        pad = L - ep - 1
        for content in product(tokens, repeat=n):
            stack = []
            ok = True
            for t in content:
                if t in open_to_close:
                    stack.append(t)
                elif t in close_to_open:
                    exp = close_to_open[t]
                    if not stack or stack[-1] != exp:
                        ok = False; break
                    stack.pop()
            if not ok or stack:
                continue
            seq = [SOS_token] + list(content) + [EOS_token] + [PAD_token]*pad
            seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs

def generate_all_sequences_for_all_languages(L):
    all_generation_functions = [gen_aNbN, gen_baN, gen_bbaN, gen_aNbNcN, gen_dyck_not_nested, gen_dyck_nested]
    
    for gen_fn in all_generation_functions:
        print(f"Generating sequences for {gen_fn.__name__} with length {L}")
        result = [] # 1. Create a list to hold the results

        all_sequences = gen_fn(L)
        for seq in all_sequences:
            result.append(seq) # 2. Append instead of yield
        
        for seq in result:
            for ch in seq:
                if ch == SOS_token:
                    print('SOS', end=' ')
                elif ch == EOS_token:
                    print('EOS', end=' ')
                elif ch == PAD_token:
                    print('PAD', end=' ')
                elif ch == MASK_token:
                    print('MASK', end=' ')
                else:
                    if gen_fn in [gen_dyck_not_nested, gen_dyck_nested]:
                        if ch == OPEN_P:
                            print('(', end=' ')
                        elif ch == CLOSE_P:
                            print(')', end=' ')
                        elif ch == OPEN_B:
                            print('[', end=' ')
                        elif ch == CLOSE_B:
                            print(']', end=' ')
                    else:
                        if ch == A:
                            print('A', end=' ')
                        elif ch == B:
                            print('B', end=' ')
                        elif ch == C:
                            print('C', end=' ')
            print() # New line after each sequence
        print(f"Total sequences generated for {gen_fn.__name__}: {len(all_sequences)}")
        
    return result # 3. Return the final list
# ─── generic test harness ─────────────────────────────────────────────────────

def run_oracle_test(oracle_fn, vocab_size, valid_seqs, atol=1e-5, label=''):
    """
    For every valid sequence and every masking pattern, compare oracle to brute-force.
    Collects all failures and reports them at the end.
    """
    failures = []
    total = 0
    oracle_total_s = 0.0
    for seq in valid_seqs:
        for masked in all_masking_patterns(seq):
            total += 1
            t0 = time.perf_counter()
            status, result = oracle_fn(masked, vocab_size)
            oracle_total_s += time.perf_counter() - t0
            bf = brute_force_marginals(masked, valid_seqs, vocab_size)

            if bf is None:
                if status is not None:
                    failures.append({
                        'seq': masked.tolist(),
                        'oracle': 'got result, expected None',
                        'bf': None,
                    })
            else:
                if status is None:
                    failures.append({
                        'seq': masked.tolist(),
                        'oracle': None,
                        'bf': bf,
                    })
                elif not torch.allclose(result, bf.float(), atol=atol):
                    failures.append({
                        'seq': masked.tolist(),
                        'oracle': result,
                        'bf': bf,
                    })

    if total:
        avg_us = oracle_total_s / total * 1e6
        print(f'\n[{label}] oracle avg: {avg_us:.1f} µs/call over {total} calls')

    if failures:
        msg = f'{label}: {len(failures)}/{total} failures\n'
        for f in failures[:5]:  # show first 5
            msg += f'  seq={f["seq"]}\n  oracle={f["oracle"]}\n  bf={f["bf"]}\n'
        pytest.fail(msg)

# ─── tests ────────────────────────────────────────────────────────────────────

def test_manual_small_sequences():
    sample_seqs = [torch.tensor([SOS_token, MASK_token, A, MASK_token, MASK_token, MASK_token, B, MASK_token, C, MASK_token, EOS_token, PAD_token, PAD_token], dtype=torch.long),]
    
    
    
class TestANBN:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_aNbN(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(aNbN_get_marginals, vocab_size=6, valid_seqs=seqs, label='aNbN')

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = aNbN_get_marginals(seq, vocab_size=6)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all(), \
                f'Unmasked seq {seq.tolist()} not deterministic'

    def test_invalid_returns_none(self):
        # SOS token at pos 0 missing
        bad = torch.tensor([A, B, EOS_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = aNbN_get_marginals(bad, vocab_size=6)
        assert status is None


class TestBaN:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_baN(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(baN_get_marginals, vocab_size=6, valid_seqs=seqs, label='baN')

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = baN_get_marginals(seq, vocab_size=6)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all()

    def test_position1_must_be_B(self):
        # position 1 = A → invalid
        seq = torch.tensor([SOS_token, A, B, B, EOS_token, PAD_token, PAD_token, PAD_token],
                           dtype=torch.long)
        status, _ = baN_get_marginals(seq, vocab_size=6)
        assert status is None

    def test_position1_masked_returns_B(self):
        # SOS MASK B EOS PAD PAD PAD PAD — position 1 is masked; oracle must assign prob 1 to B there
        seq = torch.tensor([SOS_token, MASK_token, B, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, result = baN_get_marginals(seq, vocab_size=6)
        if status is not None:
            assert abs(result[1, B].item() - 1.0) < 1e-5, \
                f'pos1 prob(B)={result[1, B].item():.4f}, expected 1.0'

    def test_odd_a_count_no_valid_completion(self):
        # SOS B A EOS PAD PAD PAD PAD — 1 A, parity=1, k=0 → impossible
        seq = torch.tensor([SOS_token, B, A, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = baN_get_marginals(seq, vocab_size=6)
        assert status is None

    @pytest.mark.parametrize('L', [6, 8])
    def test_different_lengths(self, L):
        seqs = gen_baN(L)
        if seqs:
            run_oracle_test(baN_get_marginals, vocab_size=6, valid_seqs=seqs, label=f'baN_L{L}')


class TestBBaN:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_bbaN(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(bbaN_get_marginals, vocab_size=6, valid_seqs=seqs, label='bbaN')

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = bbaN_get_marginals(seq, vocab_size=6)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all()

    def test_no_Bs_returns_none(self):
        # SOS A A EOS PAD ... — no B's → invalid for bbaN
        seq = torch.tensor([SOS_token, A, A, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = bbaN_get_marginals(seq, vocab_size=6)
        assert status is None

    def test_A_before_B_returns_none(self):
        # SOS A B EOS PAD ... — A before B → invalid
        seq = torch.tensor([SOS_token, A, B, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = bbaN_get_marginals(seq, vocab_size=6)
        assert status is None

    @pytest.mark.parametrize('L', [6, 8, 10])
    def test_different_lengths(self, L):
        seqs = gen_bbaN(L)
        if seqs:
            run_oracle_test(bbaN_get_marginals, vocab_size=6, valid_seqs=seqs, label=f'bbaN_L{L}')


class TestANBNCN:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_aNbNcN(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(aNbNcN_get_marginals, vocab_size=7, valid_seqs=seqs, label='aNbNcN')

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = aNbNcN_get_marginals(seq, vocab_size=7)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all()

    def test_wrong_order_returns_none(self):
        # SOS B A C EOS PAD ... — B before A is invalid
        seq = torch.tensor([SOS_token, B, A, C, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token, PAD_token],
                           dtype=torch.long)
        status, _ = aNbNcN_get_marginals(seq, vocab_size=7)
        assert status is None

    @pytest.mark.parametrize('L', [7, 10])
    def test_different_lengths(self, L):
        seqs = gen_aNbNcN(L)
        if seqs:
            run_oracle_test(aNbNcN_get_marginals, vocab_size=7, valid_seqs=seqs,
                            label=f'aNbNcN_L{L}')


class TestNotNestedParenthesesAndBrackets:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_dyck_not_nested(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(
            not_nested_parentheses_and_brackets_get_marginals,
            vocab_size=8, valid_seqs=seqs,
            label='not_nested_paren_bracket',
        )

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all()

    def test_mismatched_returns_none(self):
        # SOS ) ( EOS PAD ... — close before open → invalid paren
        seq = torch.tensor([SOS_token, CLOSE_P, OPEN_P, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        assert status is None


class TestParenthesesAndBrackets:
    @pytest.fixture(scope='class')
    def seqs(self, seq_length):
        return gen_dyck_nested(L=seq_length)

    def test_count(self, seqs):
        assert len(seqs) > 0

    def test_oracle_vs_brute_force(self, seqs):
        run_oracle_test(
            parentheses_and_brackets_get_marginals,
            vocab_size=8, valid_seqs=seqs,
            label='parentheses_and_brackets',
        )

    def test_fully_unmasked_is_deterministic(self, seqs):
        for seq in seqs:
            _, result = parentheses_and_brackets_get_marginals(seq, vocab_size=8)
            assert result is not None
            assert (result.max(dim=-1).values == 1.0).all()

    def test_wrong_close_returns_none(self):
        # SOS ( ] EOS PAD ... — close bracket doesn't match open paren
        seq = torch.tensor([SOS_token, OPEN_P, CLOSE_B, EOS_token,
                            PAD_token, PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        assert status is None

    def test_independent_vs_nested_differ(self):
        """
        ( ] ) [ is valid for not-nested (each subseq matched) but invalid for nested.
        Sequence: SOS ( ] ) [ EOS PAD PAD
        """
        seq = torch.tensor([SOS_token, OPEN_P, CLOSE_B, CLOSE_P, OPEN_B, EOS_token,
                            PAD_token, PAD_token], dtype=torch.long)
        s_nest, _ = parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        s_ind,  _ = not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8)
        # nested must reject; independent may accept
        assert s_nest is None


# ─── cross-grammar smoke tests ────────────────────────────────────────────────

class TestCrossGrammar:
    def test_baN_seqs_invalid_for_bbaN(self):
        """Most baN sequences (with A's interleaved in B-positions) are invalid for bbaN."""
        # SOS B A B EOS PAD ... — B A B is not b^n a^{2m} form
        seq = torch.tensor([SOS_token, B, A, B, EOS_token,
                            PAD_token, PAD_token, PAD_token], dtype=torch.long)
        status, _ = bbaN_get_marginals(seq, vocab_size=6)
        assert status is None  # A in the middle of B's is not valid for bbaN

    def test_marginals_sum_to_one_at_each_position(self):
        """At every non-SOS position, probabilities should sum to 1."""
        for oracle_fn, vocab_size, gen_fn in [
            (baN_get_marginals, 6, lambda: gen_baN(8)),
            (bbaN_get_marginals, 6, lambda: gen_bbaN(8)),
            (aNbNcN_get_marginals, 7, lambda: gen_aNbNcN(10)),
        ]:
            seqs = gen_fn()
            for seq in seqs[:3]:  # spot check 3 sequences
                masked = seq.clone()
                for p in range(1, masked.shape[0]):
                    masked[p] = MASK_token
                status, result = oracle_fn(masked, vocab_size)
                if status is not None:
                    row_sums = result.sum(dim=-1)
                    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
                        f'Row sums not 1: {row_sums.tolist()}'
