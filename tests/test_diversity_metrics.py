"""
Tests for src/diversity_metrics.py.

Coverage:
  - _strip: SOS/EOS/PAD removal
  - Universal metrics: uniqueness, duplication_rate, mean_lev_dist_normalized,
    bigram_diversity, trigram_diversity
  - _compute_universal edge cases: n=0, n=1, 2<=n<5, n>=5
  - DFA construction and coverage: baN (full/partial), bbaN
  - n-distribution: aNbN, aNbNcN
  - nm-distribution: bbaN (n_entropy, m_entropy, nm_joint_coverage)
  - Dyck structure: parentheses_and_brackets
  - compute_diversity_metrics dispatcher: all 6 grammars
  - Unknown grammar raises ValueError
"""

import math
import sys
from pathlib import Path

import pytest

# Make src importable from the tests directory
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diversity_metrics import (
    GRAMMAR_METRIC_SETS,
    _strip,
    _compute_universal,
    _build_ban_dfa,
    _build_bban_dfa,
    _run_dfa,
    compute_dfa_coverage,
    compute_n_distribution,
    compute_nm_distribution,
    compute_dyck_structure,
    compute_diversity_metrics,
    compute_diversity_distributions,
    uniqueness,
    duplication_rate,
    mean_lev_dist_normalized,
    bigram_diversity,
    trigram_diversity,
)

# ---------------------------------------------------------------------------
# Token constants (default scheme)
# ---------------------------------------------------------------------------
A, B, EOS, SOS, PAD, MASK, C = 0, 1, 2, 3, 4, 5, 6
OPEN_PAREN, CLOSE_PAREN, OPEN_BRACKET, CLOSE_BRACKET = 0, 1, 6, 7

BASE_VOCAB = {'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK, 'a': A, 'b': B}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seq(*tokens):
    """Wrap content tokens in SOS/EOS for strip testing."""
    return (SOS,) + tuple(tokens) + (EOS,)


def nan(x):
    return math.isnan(x)


# ===========================================================================
# _strip
# ===========================================================================

class TestStrip:
    def test_removes_sos_eos(self):
        seq = make_seq(A, B, A)
        assert _strip(seq, BASE_VOCAB) == (A, B, A)

    def test_no_sos(self):
        seq = (A, B, EOS)
        assert _strip(seq, BASE_VOCAB) == (A, B)

    def test_no_eos(self):
        seq = (SOS, A, B, A)
        assert _strip(seq, BASE_VOCAB) == (A, B, A)

    def test_strips_trailing_pad(self):
        seq = (SOS, A, EOS, PAD, PAD)
        assert _strip(seq, BASE_VOCAB) == (A,)

    def test_truncates_at_first_eos(self):
        seq = (SOS, A, EOS, B, A, EOS)
        assert _strip(seq, BASE_VOCAB) == (A,)

    def test_empty_content(self):
        seq = (SOS, EOS)
        assert _strip(seq, BASE_VOCAB) == ()

    def test_tensor_input(self):
        import torch
        seq = torch.tensor([SOS, A, B, EOS])
        assert _strip(seq, BASE_VOCAB) == (A, B)


# ===========================================================================
# Universal metrics (unit)
# ===========================================================================

class TestUniqueness:
    def test_all_unique(self):
        seqs = [(A,), (B,), (A, B)]
        assert uniqueness(seqs) == pytest.approx(1.0)

    def test_all_duplicate(self):
        seqs = [(A, B)] * 4
        assert uniqueness(seqs) == pytest.approx(0.25)

    def test_empty(self):
        assert nan(uniqueness([]))

    def test_single(self):
        assert uniqueness([(A,)]) == pytest.approx(1.0)


class TestDuplicationRate:
    def test_all_unique(self):
        seqs = [(A,), (B,), (A, B)]
        assert duplication_rate(seqs) == pytest.approx(0.0)

    def test_all_duplicate(self):
        seqs = [(A, B)] * 4
        assert duplication_rate(seqs) == pytest.approx(0.75)

    def test_empty(self):
        assert nan(duplication_rate([]))


class TestMeanLevDist:
    def test_identical_sequences(self):
        seqs = [(A, B, A)] * 5
        val, n_used = mean_lev_dist_normalized(seqs)
        assert val == pytest.approx(0.0)
        assert n_used == 5

    def test_single_sequence(self):
        val, n_used = mean_lev_dist_normalized([(A, B)])
        assert nan(val)
        assert n_used == 1

    def test_empty(self):
        val, n_used = mean_lev_dist_normalized([])
        assert nan(val)
        assert n_used == 0

    def test_distinct_sequences(self):
        seqs = [(A,) * 4, (B,) * 4]
        val, _ = mean_lev_dist_normalized(seqs)
        assert 0.0 < val <= 1.0

    def test_subsampling_at_200(self):
        seqs = [((A + i % 3,) * 3) for i in range(201)]
        val, n_used = mean_lev_dist_normalized(seqs)
        assert n_used == 200
        assert 0.0 <= val <= 1.0


class TestNgramDiversity:
    def test_all_same_bigrams(self):
        seqs = [(A, B, A, B)] * 3
        val = bigram_diversity(seqs)
        # Only unique bigrams: {(A,B),(B,A)} = 2; total bigrams = 3*3 = 9
        assert val == pytest.approx(2 / 9)

    def test_all_unique_bigrams(self):
        # 3 seqs each with 1 bigram and all bigrams different
        seqs = [(A, B), (B, A), (A, A)]
        assert bigram_diversity(seqs) == pytest.approx(1.0)

    def test_too_short_skipped(self):
        # Sequences shorter than 2 contribute nothing
        seqs = [(A,), (B,)]
        assert nan(bigram_diversity(seqs))

    def test_trigram_basic(self):
        seqs = [(A, B, A)]
        val = trigram_diversity(seqs)
        assert val == pytest.approx(1.0)


# ===========================================================================
# _compute_universal edge cases
# ===========================================================================

class TestComputeUniversal:
    def test_n_zero(self):
        out = _compute_universal([])
        assert out['n_correct'] == 0
        assert out['n_correct_too_low'] is True
        for k in ('uniqueness', 'duplication_rate', 'mean_lev_dist_normalized',
                  'bigram_diversity', 'trigram_diversity'):
            assert nan(out[k]), f"Expected NaN for {k} with n=0"
        assert out['lev_n_used'] == 0

    def test_n_one(self):
        stripped = [(A, B, A)]
        out = _compute_universal(stripped)
        assert out['n_correct'] == 1
        assert out['n_correct_too_low'] is True
        assert out['uniqueness'] == pytest.approx(1.0)
        assert out['duplication_rate'] == pytest.approx(0.0)
        assert nan(out['mean_lev_dist_normalized'])
        assert out['lev_n_used'] == 1
        # n-grams are computed for n==1
        assert not nan(out['bigram_diversity'])

    def test_n_two_to_four(self):
        stripped = [(A,), (B,), (A, B)]
        out = _compute_universal(stripped)  # n=3
        assert out['n_correct'] == 3
        assert out['n_correct_too_low'] is True
        for k in ('uniqueness', 'duplication_rate', 'mean_lev_dist_normalized',
                  'bigram_diversity', 'trigram_diversity'):
            assert nan(out[k]), f"Expected NaN for {k} with n=3"

    def test_n_five(self):
        stripped = [(A,), (B,), (A, B), (B, A), (A, A)]
        out = _compute_universal(stripped)
        assert out['n_correct'] == 5
        assert out['n_correct_too_low'] is False
        assert not nan(out['uniqueness'])
        assert not nan(out['mean_lev_dist_normalized'])
        assert not nan(out['bigram_diversity'])


# ===========================================================================
# DFA construction and coverage
# ===========================================================================

class TestBanDFA:
    def setup_method(self):
        self.dfa = _build_ban_dfa()

    def test_states_and_transitions(self):
        assert len(self.dfa['states']) == 4
        assert len(self.dfa['transitions']) == 8
        assert self.dfa['initial'] == 0
        assert 1 in self.dfa['accepting']

    def test_accepts_BA(self):
        # B A A = B A^2  (k=1) → accepting
        visited, taken = _run_dfa(self.dfa, (B, A, A))
        assert 1 in visited  # q_even visited

    def test_accepts_B_only(self):
        # B A^0 (k=0) — starts and ends in q_even
        visited, taken = _run_dfa(self.dfa, (B,))
        assert 1 in visited

    def test_rejects_A_first(self):
        visited, taken = _run_dfa(self.dfa, (A, B))
        # Should land in dead state
        assert 3 in visited

    def test_full_state_and_transition_coverage(self):
        # Sequences chosen to trigger all 8 transitions (and thus all 4 states):
        #   (A,A)    : (0,A)→dead, (3,A)→dead  — covers (0,0),(3,0)
        #   (A,B)    : (0,A)→dead, (3,B)→dead  — covers (0,0),(3,1)
        #   (B,B)    : (0,B)→even, (1,B)→dead  — covers (0,1),(1,1)
        #   (B,A,A)  : (0,B)→even, (1,A)→odd, (2,A)→even — covers (0,1),(1,0),(2,0)
        #   (B,A,B)  : (0,B)→even, (1,A)→odd, (2,B)→dead — covers (0,1),(1,0),(2,1)
        seqs = [(A, A), (A, B), (B, B), (B, A, A), (B, A, B)]
        cov = compute_dfa_coverage('baN', seqs)
        assert cov['dfa_state_coverage'] == pytest.approx(1.0)
        assert cov['dfa_transition_coverage'] == pytest.approx(1.0)

    def test_partial_coverage(self):
        # Only (B,) → visits q_start and q_even (0,1); takes (0,1)→1
        seqs = [(B,)]
        cov = compute_dfa_coverage('baN', seqs)
        assert cov['dfa_state_coverage'] < 1.0
        assert cov['dfa_transition_coverage'] < 1.0


class TestBbanDFA:
    def setup_method(self):
        self.dfa = _build_bban_dfa()

    def test_states_and_transitions(self):
        assert len(self.dfa['states']) == 5
        assert len(self.dfa['transitions']) == 10
        assert self.dfa['initial'] == 0
        assert frozenset([1, 3]) == self.dfa['accepting']

    def test_accepts_B_only(self):
        visited, _ = _run_dfa(self.dfa, (B,))
        assert 1 in visited  # q_b is accepting

    def test_accepts_BB_AA(self):
        visited, _ = _run_dfa(self.dfa, (B, B, A, A))
        assert 3 in visited  # q_a_even is accepting

    def test_rejects_A_first(self):
        visited, _ = _run_dfa(self.dfa, (A,))
        assert 4 in visited  # q_dead

    def test_full_coverage_bbaN(self):
        # Cover all 5 states and 10 transitions
        seqs = [
            (B,),        # q_start→q_b
            (B, A),      # +q_a_odd
            (B, A, A),   # +q_a_even
            (A,),        # q_start→q_dead
        ]
        cov = compute_dfa_coverage('bbaN', seqs)
        assert cov['dfa_state_coverage'] == pytest.approx(1.0)


# ===========================================================================
# n-distribution
# ===========================================================================

ANBN_VOCAB = {
    'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
    'a': A, 'b': B,
    'valid_n_range': range(1, 9),  # L=16: n in [1..8]
}


class TestNDistribution:
    def test_single_n_value(self):
        # All sequences have n=2 (2 A's)
        seqs = [(A, A, B, B)] * 5
        out = compute_n_distribution('aNbN', seqs, ANBN_VOCAB)
        assert out['n_entropy'] == pytest.approx(0.0)  # no uncertainty
        assert 0.0 < out['n_coverage'] <= 1.0

    def test_full_n_coverage(self):
        # One sequence per valid n value
        seqs = [tuple([A] * n + [B] * n) for n in range(1, 9)]
        out = compute_n_distribution('aNbN', seqs, ANBN_VOCAB)
        assert out['n_coverage'] == pytest.approx(1.0)
        assert out['n_entropy'] > 0.0

    def test_empty_sequences(self):
        out = compute_n_distribution('aNbN', [], ANBN_VOCAB)
        assert nan(out['n_entropy'])
        assert nan(out['n_coverage'])


# ===========================================================================
# nm-distribution (bbaN)
# ===========================================================================

BBAN_VOCAB = {
    'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
    'a': A, 'b': B,
    'valid_nm_pairs': frozenset(
        (n, m) for n in range(1, 5) for m in range(0, 5)
    ),
}


class TestNmDistribution:
    def test_basic(self):
        # B^2 A^4: n=2, m=2
        seqs = [(B, B, A, A, A, A)] * 5
        out = compute_nm_distribution(seqs, BBAN_VOCAB)
        assert out['n_entropy'] == pytest.approx(0.0)
        assert out['m_entropy'] == pytest.approx(0.0)
        assert 0.0 < out['nm_joint_coverage'] <= 1.0

    def test_varied_nm(self):
        seqs = [
            (B, A, A),         # n=1, m=1
            (B, B, A, A, A, A),  # n=2, m=2
            (B,),              # n=1, m=0
        ]
        out = compute_nm_distribution(seqs, BBAN_VOCAB)
        assert out['n_entropy'] > 0.0
        assert out['m_entropy'] > 0.0
        assert 0.0 < out['nm_joint_coverage'] <= 1.0

    def test_empty(self):
        out = compute_nm_distribution([], BBAN_VOCAB)
        assert nan(out['n_entropy'])
        assert nan(out['m_entropy'])
        assert nan(out['nm_joint_coverage'])


# ===========================================================================
# Dyck structure
# ===========================================================================

DYCK_VOCAB = {
    'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
    'paren_open': OPEN_PAREN,
    'paren_close': CLOSE_PAREN,
    'bracket_open': OPEN_BRACKET,
    'bracket_close': CLOSE_BRACKET,
}


class TestDyckStructure:
    def test_balanced_parens_only(self):
        # () = open-close pair → max_depth=1, L/2=1 → ratio=1.0
        seqs = [(OPEN_PAREN, CLOSE_PAREN)]
        out = compute_dyck_structure(seqs, DYCK_VOCAB)
        assert out['max_depth_ratio_mean'] == pytest.approx(1.0)
        assert out['n_zero_paren_sequences'] == 0

    def test_brackets_parens_ratio(self):
        # ( [ ] ) → 1 paren, 1 bracket → ratio=1.0
        seqs = [(OPEN_PAREN, OPEN_BRACKET, CLOSE_BRACKET, CLOSE_PAREN)]
        out = compute_dyck_structure(seqs, DYCK_VOCAB)
        assert out['brackets_parens_ratio_mean'] == pytest.approx(1.0)

    def test_zero_paren_counted(self):
        # [ ] only — no parens → n_zero_paren_sequences += 1
        seqs = [(OPEN_BRACKET, CLOSE_BRACKET)]
        out = compute_dyck_structure(seqs, DYCK_VOCAB)
        assert out['n_zero_paren_sequences'] == 1
        assert nan(out['brackets_parens_ratio_mean'])  # no BP ratios to mean

    def test_empty(self):
        out = compute_dyck_structure([], DYCK_VOCAB)
        assert nan(out['max_depth_ratio_mean'])
        assert out['n_zero_paren_sequences'] == 0


# ===========================================================================
# compute_diversity_metrics dispatcher
# ===========================================================================

def _make_stripped_seqs(raw_tuples):
    """Return already-stripped tuples (no SOS/EOS) wrapped as if from sequences."""
    return list(raw_tuples)


class TestDispatcher:
    """Test that compute_diversity_metrics returns the right keys for each grammar."""

    EXPECTED_KEYS = [
        'n_correct', 'n_correct_too_low', 'uniqueness', 'duplication_rate',
        'mean_lev_dist_normalized', 'lev_n_used', 'bigram_diversity', 'trigram_diversity',
        'dfa_state_coverage', 'dfa_transition_coverage',
        'n_entropy', 'n_coverage', 'm_entropy', 'nm_joint_coverage',
        'max_depth_ratio_mean', 'max_depth_ratio_std',
        'brackets_parens_ratio_mean', 'brackets_parens_ratio_std',
        'n_zero_paren_sequences',
    ]

    def _check_keys(self, out):
        for k in self.EXPECTED_KEYS:
            assert k in out, f"Missing key: {k}"

    def test_baN_with_zero_correct(self):
        vocab = {**BASE_VOCAB}
        out = compute_diversity_metrics('baN', [], vocab)
        self._check_keys(out)
        assert out['n_correct'] == 0
        # DFA-specific fields should be NaN when n=0
        assert nan(out['dfa_state_coverage'])
        assert nan(out['dfa_transition_coverage'])
        # Non-applicable fields are NaN
        assert nan(out['n_entropy'])
        assert nan(out['max_depth_ratio_mean'])

    def test_baN_with_sequences(self):
        vocab = {**BASE_VOCAB}
        # 5 correct B A^{2k} sequences (no SOS/EOS — they're pre-stripped by the grammar)
        seqs = [make_seq(B, A, A)] * 5  # B A^2 with SOS/EOS
        out = compute_diversity_metrics('baN', seqs, vocab)
        self._check_keys(out)
        assert out['n_correct'] == 5
        assert not nan(out['dfa_state_coverage'])
        assert nan(out['n_entropy'])
        assert nan(out['max_depth_ratio_mean'])

    def test_bbaN(self):
        vocab = {**BASE_VOCAB, 'valid_nm_pairs': frozenset(
            (n, m) for n in range(1, 4) for m in range(0, 4)
        )}
        seqs = [make_seq(B, A, A)] * 5
        out = compute_diversity_metrics('bbaN', seqs, vocab)
        self._check_keys(out)
        assert not nan(out['dfa_state_coverage'])
        assert not nan(out['n_entropy'])   # n_entropy from nm_distribution
        assert not nan(out['m_entropy'])
        assert nan(out['n_coverage'])      # n_coverage only for L3/L5

    def test_aNbN(self):
        vocab = {**BASE_VOCAB, 'valid_n_range': range(1, 5)}
        seqs = [make_seq(A, A, B, B)] * 5
        out = compute_diversity_metrics('aNbN', seqs, vocab)
        self._check_keys(out)
        assert nan(out['dfa_state_coverage'])
        assert not nan(out['n_entropy'])
        assert not nan(out['n_coverage'])
        assert nan(out['m_entropy'])
        assert nan(out['nm_joint_coverage'])
        assert nan(out['max_depth_ratio_mean'])

    def test_aNbNcN(self):
        vocab = {**BASE_VOCAB, 'c': C, 'valid_n_range': range(1, 4)}
        seqs = [make_seq(A, B, C)] * 5
        out = compute_diversity_metrics('aNbNcN', seqs, vocab)
        self._check_keys(out)
        assert nan(out['dfa_state_coverage'])
        assert not nan(out['n_entropy'])

    def test_parentheses_and_brackets(self):
        vocab = {
            'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
            'paren_open': OPEN_PAREN, 'paren_close': CLOSE_PAREN,
            'bracket_open': OPEN_BRACKET, 'bracket_close': CLOSE_BRACKET,
        }
        seqs = [make_seq(OPEN_PAREN, CLOSE_PAREN)] * 5
        out = compute_diversity_metrics('parentheses_and_brackets', seqs, vocab)
        self._check_keys(out)
        assert nan(out['dfa_state_coverage'])
        assert nan(out['n_entropy'])
        assert not nan(out['max_depth_ratio_mean'])

    def test_not_nested_parentheses_and_brackets(self):
        vocab = {
            'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
            'paren_open': OPEN_PAREN, 'paren_close': CLOSE_PAREN,
            'bracket_open': OPEN_BRACKET, 'bracket_close': CLOSE_BRACKET,
        }
        seqs = [make_seq(OPEN_PAREN, CLOSE_PAREN)] * 5
        out = compute_diversity_metrics('not_nested_parentheses_and_brackets', seqs, vocab)
        self._check_keys(out)
        assert nan(out['dfa_state_coverage'])
        assert not nan(out['max_depth_ratio_mean'])

    def test_unknown_grammar_raises(self):
        with pytest.raises(ValueError, match="Unknown grammar"):
            compute_diversity_metrics('bogus', [], BASE_VOCAB)


# ===========================================================================
# compute_diversity_distributions
# ===========================================================================

class TestDiversityDistributions:
    def test_baN_returns_empty_dict(self):
        out = compute_diversity_distributions('baN', [], BASE_VOCAB)
        assert isinstance(out, dict)
        # baN has no distribution sidefile
        assert 'n_values' not in out

    def test_aNbN_returns_n_values(self):
        vocab = {**BASE_VOCAB, 'valid_n_range': range(1, 5)}
        seqs = [make_seq(A, A, B, B), make_seq(A, B)]
        out = compute_diversity_distributions('aNbN', seqs, vocab)
        assert 'n_values' in out
        assert out['n_values'] == [2, 1]  # A count per seq

    def test_bbaN_returns_nm_values(self):
        vocab = {**BASE_VOCAB, 'valid_nm_pairs': frozenset()}
        seqs = [make_seq(B, A, A)]  # n=1, m=1
        out = compute_diversity_distributions('bbaN', seqs, vocab)
        assert 'n_values' in out
        assert 'm_values' in out
        assert out['n_values'] == [1]
        assert out['m_values'] == [1]

    def test_dyck_returns_depth_and_bp(self):
        vocab = {
            'sos': SOS, 'eos': EOS, 'pad': PAD, 'mask': MASK,
            'paren_open': OPEN_PAREN, 'paren_close': CLOSE_PAREN,
            'bracket_open': OPEN_BRACKET, 'bracket_close': CLOSE_BRACKET,
        }
        seqs = [make_seq(OPEN_PAREN, CLOSE_PAREN)]
        out = compute_diversity_distributions('parentheses_and_brackets', seqs, vocab)
        assert 'max_depth_ratios' in out
        assert 'brackets_parens_ratios' in out
        assert 'n_correct' in out

    def test_unknown_grammar_raises(self):
        with pytest.raises(ValueError):
            compute_diversity_distributions('bogus', [], BASE_VOCAB)


# ===========================================================================
# GRAMMAR_METRIC_SETS completeness
# ===========================================================================

class TestGrammarMetricSets:
    def test_all_six_grammars_present(self):
        expected = {
            'baN', 'bbaN', 'aNbN',
            'parentheses_and_brackets', 'aNbNcN',
            'not_nested_parentheses_and_brackets',
        }
        assert set(GRAMMAR_METRIC_SETS.keys()) == expected

    def test_all_include_universal(self):
        for name, sets in GRAMMAR_METRIC_SETS.items():
            assert 'universal' in sets, f"'universal' missing for {name}"
