"""
tests/test_grammar_alignment.py
================================
Verifies that for every L1–L6 grammar, all encodings agree:

  (a) re_data.grammar_rules                (the ground-truth lambda)
  (b) REGrammar's rule1 AND rule2          (the wrapper used by eval scripts)
  (c) diversity_metrics DFA acceptance     (where applicable: L1, L2)
  (d) valid_n_range / valid_nm_pairs       (where applicable: L2, L3, L5)

Each grammar has a small hand-picked test set: 4-8 positive examples (in the
language) and 4-8 negative examples (not in the language). For every example,
all encodings must produce the same accept/reject decision.

Run from project root:
    pytest tests/test_grammar_alignment.py -v

This test exists to prevent silent drift between the four encodings. If you
modify any of (re_data, re_grammar, diversity_metrics' DFAs, valid ranges),
running this test will catch disagreements before they pollute results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import pytest
import torch

from datasets.constants import EOS_token, SOS_token, PAD_token
from datasets.re_grammar import REGrammar
from datasets import re_data as rd

# Canonical rule check from re_data — duplicate here so the test is independent
# of any wrapper code. If this disagrees with re_data.grammar_rules, this test
# is the wrong one; update it.
def canonical_rule(grammar_name):
    rules = {
        'baN':    lambda x: rd.check_even_number_of_as(x) and rd.check_begins_with_b(x),
        'bbaN':   lambda x: rd.check_even_number_of_as_end(x) and rd.check_bs_before_as(x),
        'aNbN':   lambda x: rd.check_same_number_as_bs(x) and rd.check_as_before_bs(x),
        'parentheses_and_brackets':
                  lambda x: rd.check_matched_parentheses_and_brackets(x),
        'aNbNcN': lambda x: rd.check_same_number_as_bs_cs(x)
                            and rd.check_as_before_bs_before_cs(x),
        'not_nested_parentheses_and_brackets':
                  lambda x: rd.check_matched_parentheses(x) and rd.check_matched_brackets(x),
    }
    return rules[grammar_name]


# ---------------------------------------------------------------------------
# Token constants (project scheme)
# ---------------------------------------------------------------------------
SOS, EOS, PAD = 3, 2, 4
A, B, C = 0, 1, 6
LP, RP = 0, 1   # parens: open=0, close=1
LB, RB = 6, 7   # brackets: open=6, close=7


def _seq(content):
    """Wrap a content-token list into a full sequence: SOS + content + EOS."""
    return torch.tensor([SOS] + list(content) + [EOS], dtype=torch.long)


# ---------------------------------------------------------------------------
# Test cases per grammar
# ---------------------------------------------------------------------------
# Each entry is (content_tokens, expected_in_language).
# Content tokens exclude SOS/EOS. The _seq helper adds them.

TEST_CASES = {
    'baN': [
        # Positive: starts with b, total #a even
        ([B],                                            True),   # just b: 0 a's
        ([B, A, A],                                      True),   # b aa
        ([B, A, A, A, A],                                True),   # b a^4
        ([B, A, A, B, A, A],                             True),   # b aa b aa — multiple b's, even a's
        ([B, B, A, A],                                   True),   # bb a^2 — multiple b's at start
        ([B, B],                                         True),   # bb: 0 a's
        # Negative
        ([B, A],                                         False),  # 1 a, odd
        ([B, A, A, A],                                   False),  # 3 a's, odd
        ([A, A],                                         False),  # starts with a
        ([A, B, A, A],                                   False),  # starts with a
        ([B, A, B, A, A],                                False),  # 3 a's, odd (despite non-initial b)
    ],

    'bbaN': [
        # Positive: all b's before any a; #a-after-last-b is even (incl. 0)
        ([B],                                            True),   # just b: m=0
        ([B, B],                                         True),   # bb: m=0
        ([B, A, A],                                      True),   # b aa: n=1, m=1
        ([B, B, A, A],                                   True),   # bb aa: n=2, m=1
        ([B, A, A, A, A],                                True),   # b aaaa: n=1, m=2
        # Negative
        ([B, A],                                         False),  # 1 a, odd
        ([B, A, B, A],                                   False),  # b after a (rule2 fail)
        ([A, A],                                         False),  # no b's at all (rule2 fail)
        ([A, B, A],                                      False),  # a before b
        ([B, A, A, A],                                   False),  # 3 a's, odd
    ],

    'aNbN': [
        # Positive: a^n b^n
        ([A, B],                                         True),   # n=1
        ([A, A, B, B],                                   True),   # n=2
        ([A, A, A, B, B, B],                             True),   # n=3
        # Negative
        ([A, B, A, B],                                   False),  # interleaved
        ([A, A, B],                                      False),  # n_a != n_b
        ([A, B, B],                                      False),  # n_a != n_b
        ([B, A],                                         False),  # b before a
        ([A, A, A, B, B],                                False),  # n_a != n_b
    ],

    'parentheses_and_brackets': [   # L4 = strict Dyck (joint stack)
        # Positive: properly nested
        ([LP, RP],                                       True),   # ()
        ([LB, RB],                                       True),   # []
        ([LP, LB, RB, RP],                               True),   # ([])
        ([LP, RP, LB, RB],                               True),   # ()[]
        ([LB, LP, LP, RP, RP, RB],                       True),   # [(())]
        # Negative
        ([LP, LB, RP, RB],                               False),  # ([)] — CRITICAL: must fail
        ([LB, LP, RB, RP],                               False),  # [(]) — CRITICAL: must fail
        ([LP, RB],                                       False),  # mismatched
        ([LP],                                           False),  # unbalanced
        ([RP, LP],                                       False),  # close before open
    ],

    'aNbNcN': [
        # Positive: a^n b^n c^n
        ([A, B, C],                                      True),   # n=1
        ([A, A, B, B, C, C],                             True),   # n=2
        ([A, A, A, B, B, B, C, C, C],                    True),   # n=3
        # Negative
        ([A, B, C, A, B, C],                             False),  # interleaved
        ([A, A, B, C, C],                                False),  # counts mismatched
        ([A, B, B, C],                                   False),  # counts mismatched
        ([A, C, B],                                      False),  # order wrong
        ([B, A, B, C],                                   False),  # order wrong
    ],

    'not_nested_parentheses_and_brackets': [   # L6 = each type matched independently
        # Positive: parens match AND brackets match (each type ignores the other)
        ([LP, RP],                                       True),   # ()
        ([LB, RB],                                       True),   # []
        ([LP, LB, RP, RB],                               True),   # ([)]  — non-nested but L6-valid
        ([LB, LP, RB, RP],                               True),   # [(]) — non-nested but L6-valid
        ([LP, LB, RB, RP],                               True),   # ([])  — nested also OK
        # Negative
        ([LP, RB],                                       False),  # mismatched type
        ([LP, LP, RP],                                   False),  # parens unbalanced
        ([LB],                                           False),  # bracket unclosed
        ([RP, LP],                                       False),  # close before open
    ],
}


# ---------------------------------------------------------------------------
# Test: re_data canonical vs REGrammar wrapper agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('grammar_name', sorted(TEST_CASES.keys()))
def test_re_grammar_matches_canonical(grammar_name):
    """REGrammar's (rule1 AND rule2) must equal re_data's canonical rule."""
    g = REGrammar(grammar_name=grammar_name, l=32)
    canonical = canonical_rule(grammar_name)

    for content, expected in TEST_CASES[grammar_name]:
        seq = _seq(content)
        wrapper = g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq)
        truth = bool(canonical(seq))
        assert wrapper == expected == truth, (
            f"\n{grammar_name}: {content}\n"
            f"  expected_in_language = {expected}\n"
            f"  canonical re_data    = {truth}\n"
            f"  REGrammar wrapper    = {wrapper}\n"
            f"  rule1 = {g.does_satisfy_rule1(seq)}, "
            f"rule2 = {g.does_satisfy_rule2(seq)}\n"
            "If these disagree, _GRAMMAR_RULES in re_grammar.py does not "
            "match re_data.grammar_rules. Update it."
        )


# ---------------------------------------------------------------------------
# Test: DFA acceptance matches the rule check (L1, L2 only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('grammar_name', ['baN', 'bbaN'])
def test_dfa_matches_rule_check(grammar_name):
    """For grammars with DFA coverage, DFA acceptance must equal rule check."""
    from diversity_metrics import _get_dfa, _run_dfa

    g = REGrammar(grammar_name=grammar_name, l=32)
    vocab = g.vocab_info()
    dfa = _get_dfa(grammar_name)

    for content, expected in TEST_CASES[grammar_name]:
        seq = _seq(content)
        # The DFA runs on stripped content only.
        rule_accept = g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq)
        visited, _ = _run_dfa(dfa, tuple(content))
        # Final state is the last visited state along the trajectory.
        state = dfa['initial']
        for token in content:
            nxt = dfa['transitions'].get((state, int(token)))
            if nxt is None:
                state = None
                break
            state = nxt
        dfa_accept = state in dfa['accepting']

        assert rule_accept == dfa_accept == expected, (
            f"\n{grammar_name}: {content}\n"
            f"  expected      = {expected}\n"
            f"  rule check    = {rule_accept}\n"
            f"  DFA accept    = {dfa_accept}\n"
            "If these disagree, the DFA in diversity_metrics.py does not "
            "match the rule check. Update the DFA."
        )


# ---------------------------------------------------------------------------
# Test: valid_n_range / valid_nm_pairs include all sequences the rule accepts
# ---------------------------------------------------------------------------

def test_valid_n_range_aNbN():
    """For every n in valid_n_range, a^n b^n is rule-correct."""
    g = REGrammar(grammar_name='aNbN', l=32)
    for n in g.valid_n_range():
        seq = _seq([A] * n + [B] * n)
        assert g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq), (
            f"aNbN n={n} from valid_n_range is rejected by rule check"
        )


def test_valid_n_range_aNbNcN():
    """For every n in valid_n_range, a^n b^n c^n is rule-correct."""
    g = REGrammar(grammar_name='aNbNcN', l=32)
    for n in g.valid_n_range():
        seq = _seq([A] * n + [B] * n + [C] * n)
        assert g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq), (
            f"aNbNcN n={n} from valid_n_range is rejected by rule check"
        )


def test_valid_nm_pairs_bbaN():
    """For every (n, m) in valid_nm_pairs, b^n a^{2m} is rule-correct.

    Also: m=0 must be included (was missing pre-fix).
    """
    g = REGrammar(grammar_name='bbaN', l=32)
    pairs = g.bbaN_valid_nm_pairs()

    # Spot-check inclusion of m=0 cases (the bug we just fixed)
    assert any(m == 0 for (n, m) in pairs), (
        "bbaN_valid_nm_pairs excludes all m=0 cases, but canonical bbaN "
        "accepts pure-b sequences (e.g., 'b' has n=1, m=0)."
    )

    for (n, m) in pairs:
        seq = _seq([B] * n + [A] * (2 * m))
        assert g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq), (
            f"bbaN (n={n}, m={m}) from valid_nm_pairs is rejected by rule check"
        )


# ---------------------------------------------------------------------------
# Regression test: the two bugs that motivated this audit
# ---------------------------------------------------------------------------

def test_L4_rejects_non_nested_dyck():
    """L4 (parentheses_and_brackets) must reject `([)]` and `[(])`.

    Pre-fix, the wrapper used two independent paren/bracket checks, which both
    pass on these strings. The canonical joint-stack check rejects them.
    """
    g = REGrammar(grammar_name='parentheses_and_brackets', l=32)

    for content in [[LP, LB, RP, RB], [LB, LP, RB, RP]]:
        seq = _seq(content)
        accept = g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq)
        assert not accept, (
            f"L4 (Dyck) incorrectly accepts non-nested {content}. "
            "This means re_grammar.py is using the old buggy mapping "
            "(two independent checks) instead of "
            "check_matched_parentheses_and_brackets."
        )


def test_L1_accepts_multiple_bs():
    """L1 (baN) must accept sequences with non-initial b's like `b a a b a a`.

    Pre-fix, the DFA routed non-initial b's to a dead state. The rule check
    accepted these sequences; this test ensures the DFA does too.
    """
    g = REGrammar(grammar_name='baN', l=32)
    seq = _seq([B, A, A, B, A, A])  # starts with b, 4 a's (even)
    assert g.does_satisfy_rule1(seq) and g.does_satisfy_rule2(seq)

    from diversity_metrics import _get_dfa
    dfa = _get_dfa('baN')
    state = dfa['initial']
    for token in [B, A, A, B, A, A]:
        state = dfa['transitions'].get((state, token))
    assert state in dfa['accepting'], (
        "baN DFA rejects 'b a a b a a', which is rule-correct. "
        "The DFA's non-initial-b transitions must self-loop, not route to dead."
    )


def test_L2_rule1_is_check_even_number_of_as_end():
    """L2 (bbaN) rule1 must be check_even_number_of_as_end, not check_even_number_of_as.

    These differ on sequences with non-monotonic structure. For sequences
    that pass rule2 (b's before a's), they agree, so this test exercises the
    case where they would differ.
    """
    g = REGrammar(grammar_name='bbaN', l=32)
    # Sequence that fails rule2 but where total-a-count is even and trailing-a-count is odd:
    # e.g., 'b a b a a a' — 4 a's total (even), trailing run is 'aaa' (odd, but rule2 fails too)
    # Better case: a sequence where total-a is even but trailing-a is odd.
    # 'a a b a a a' — fails rule2; total a=5 (odd), trailing a=3 (odd). No good.
    # Easier: rely on the function identity check.
    assert g._rule1_fn is rd.check_even_number_of_as_end, (
        f"bbaN rule1 is {g._rule1_fn.__name__}, expected check_even_number_of_as_end. "
        "Update _GRAMMAR_RULES['bbaN'] in re_grammar.py."
    )
