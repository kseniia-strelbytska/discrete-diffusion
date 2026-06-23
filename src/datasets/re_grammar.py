## Taken from https://github.com/meszarosanna/rule_extrapolation/tree/main/rule_extrapolation

"""
REGrammar: wraps RE_data generators into the FormalGrammar interface.

Grammar rule definitions are taken VERBATIM from `re_data.grammar_rules`, which
is the canonical source of truth. The mapping below uses the same check
functions that re_data's `grammar_rules` lambda uses, in the same composition.
If you ever update this file, update it from re_data.grammar_rules — not from
notes, not from the table below.

Project token scheme (matches constants.py and re_data.py):
  SOS=3, EOS=2, PAD=4, MASK=5
  A=0, B=1, C=6
  open_paren=0, close_paren=1, open_bracket=6, close_bracket=7

Grammar vocab_sizes:
  aNbN, abN, baN, bbaN, aNbM, aNbNaN  → vocab_size=6  (A=0, B=1)
  parentheses                          → vocab_size=6  (open=0, close=1)
  aNbNcN                               → vocab_size=7  (A=0, B=1, C=6)
  brackets                             → vocab_size=8  (open=6, close=7)
  parentheses_and_brackets,
  separated_parentheses_and_brackets,
  not_nested_parentheses_and_brackets  → vocab_size=8  (open_p=0, close_p=1,
                                                         open_b=6, close_b=7)

Canonical rule mapping per re_data.grammar_rules:

  L1 = baN
       regular: starts with b AND total #a is even (multiple b's allowed)
       rule1 = check_even_number_of_as
       rule2 = check_begins_with_b

  L2 = bbaN
       regular: all b's before any a (n>=1 b's), #a-after-last-b is even
       rule1 = check_even_number_of_as_end   ← NOT check_even_number_of_as
       rule2 = check_bs_before_as

  L3 = aNbN
       context-free: a^n b^n
       rule1 = check_same_number_as_bs
       rule2 = check_as_before_bs

  L4 = parentheses_and_brackets
       context-free: strict Dyck over {(), []} with joint nesting
       rule1 = check_matched_parentheses_and_brackets   ← SINGLE joint-stack check
       rule2 = (always True)
       NOTE: re_data exposes L4 as a single combined check. The two-rule
       framework collapses for this grammar — there is no decomposition into
       "matched ( )" and "matched [ ]" that reproduces the joint-stack semantics
       (e.g., '( [ ) ]' satisfies both bracket-only and paren-only checks
       independently but fails the joint Dyck check). Reporting rule2=True for
       this grammar is the canonical-faithful choice; do not "decompose" L4
       without adding new check functions to re_data.

  L5 = aNbNcN
       context-sensitive: a^n b^n c^n
       rule1 = check_same_number_as_bs_cs
       rule2 = check_as_before_bs_before_cs

  L6 = not_nested_parentheses_and_brackets
       context-sensitive: parens and brackets each independently matched
       rule1 = check_matched_parentheses    ← parens, ignoring brackets
       rule2 = check_matched_brackets       ← brackets, ignoring parens
       Order matches re_data's `grammar_rules` lambda for consistency.
"""

import torch
import numpy as np
from .formal_grammar import FormalGrammar
from .constants import EOS_token, SOS_token, PAD_token, MASK_token
from . import re_data as rd


_VOCAB_SIZES = {
    'aNbN': 6, 'abN': 6, 'baN': 6, 'bbaN': 6, 'aNbM': 6, 'aNbNaN': 6,
    'aNbNcN': 7,
    'parentheses': 6,
    'brackets': 8,
    'parentheses_and_brackets': 8,
    'separated_parentheses_and_brackets': 8,
    'not_nested_parentheses_and_brackets': 8,
}

# Maps each grammar to (rule1_fn, rule2_fn). The conjunction
# rule1 AND rule2 AND format must equal re_data.grammar_rules[grammar] AND format
# for every sequence. Each entry here must match re_data.grammar_rules exactly.
_GRAMMAR_RULES = {
    # ── L1–L6: canonical-aligned (see docstring) ────────────────────────────
    'baN':    (rd.check_even_number_of_as,     rd.check_begins_with_b),
    'bbaN':   (rd.check_even_number_of_as_end, rd.check_bs_before_as),
    'aNbN':   (rd.check_same_number_as_bs,     rd.check_as_before_bs),
    'parentheses_and_brackets': (
        rd.check_matched_parentheses_and_brackets, lambda _: True,
    ),
    'aNbNcN': (rd.check_same_number_as_bs_cs,  rd.check_as_before_bs_before_cs),
    'not_nested_parentheses_and_brackets': (
        rd.check_matched_parentheses, rd.check_matched_brackets,
    ),

    # ── other supported grammars (single-rule, rule2 trivially True) ────────
    'abN':         (rd.check_same_number_as_bs, lambda _: True),
    'aNbM':        (rd.check_as_before_bs,      lambda _: True),
    'aNbNaN': (
        lambda x: rd.check_twice_many_as_than_bs(x)
                  and rd.check_bs_in_the_middle(x)
                  and rd.check_bs_together(x),
        lambda _: True,
    ),
    'parentheses': (rd.check_matched_parentheses, lambda _: True),
    'brackets':    (rd.check_matched_brackets,    lambda _: True),
    'separated_parentheses_and_brackets': (
        rd.check_separated_brackets_and_parentheses, lambda _: True,
    ),
}


def _generate_re_sequences(grammar_name, l):
    """Call the appropriate RE generator; returns list of numpy arrays in project token scheme."""
    num_samples = 2000  # used for stochastic grammars

    if grammar_name == 'aNbN':
        return rd.generate_aNbN_grammar_data(num_samples, l, all_sequences=True)
    elif grammar_name == 'aNbNaN':
        return rd.generate_aNbNaN_grammar_data(num_samples, l, all_sequences=True)
    elif grammar_name == 'aNbNcN':
        return rd.generate_aNbNcN_grammar_data(num_samples, l, all_sequences=True)
    elif grammar_name == 'abN':
        return rd.generate_abN_grammar_data(num_samples, l)
    elif grammar_name == 'baN':
        return rd.generate_baN_grammar_data(num_samples, l)
    elif grammar_name == 'bbaN':
        return rd.generate_bbaN_grammar_data(num_samples, l)
    elif grammar_name == 'aNbM':
        return rd.generate_aNbM_grammar_data(num_samples, l)
    elif grammar_name == 'brackets':
        return rd.generate_matched_brackets_data(num_samples, l)
    elif grammar_name == 'parentheses':
        return rd.generate_matched_parentheses_data(num_samples, l)
    elif grammar_name == 'parentheses_and_brackets':
        return rd.generate_matched_parentheses_and_brackets_data(num_samples, l)
    elif grammar_name == 'separated_parentheses_and_brackets':
        return rd.generate_matched_parentheses_and_matched_brackets_data(num_samples, l)
    elif grammar_name == 'not_nested_parentheses_and_brackets':
        return rd.generate_not_nested_matched_parentheses_and_brackets_data(num_samples, l)
    else:
        raise ValueError(f"Unknown RE grammar: {grammar_name}")


class REGrammar(FormalGrammar):
    """
    Wraps any RE_data grammar into the FormalGrammar interface.

    All sequences use the project token scheme (SOS=3, EOS=2, PAD=4, MASK=5)
    with content tokens starting at 0. No remapping layer is needed.

    Each grammar exposes two independent structural rules (rule1, rule2) per
    the canonical mapping in _GRAMMAR_RULES, which mirrors re_data.grammar_rules.
    """

    SUPPORTED = set(_VOCAB_SIZES.keys())

    def __init__(self, grammar_name: str, l: int):
        if grammar_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown RE grammar '{grammar_name}'. "
                f"Supported: {sorted(self.SUPPORTED)}"
            )
        super().__init__(l)
        self.grammar_name = grammar_name
        self.vocab_size = _VOCAB_SIZES[grammar_name]
        self.data = None
        self.default_eval_type = 'next_token'
        self._rule1_fn, self._rule2_fn = _GRAMMAR_RULES[grammar_name]

    # ------------------------------------------------------------------ #
    # FormalGrammar interface                                              #
    # ------------------------------------------------------------------ #

    def does_satisfy_rule1(self, seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        try:
            return bool(self._rule1_fn(seq))
        except Exception:
            return False

    def does_satisfy_rule2(self, seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        try:
            return bool(self._rule2_fn(seq))
        except Exception:
            return False

    def does_satisfy_format(self, seq):
        # Universal, grammar-independent structural format check. A correctly
        # formatted, fully-denoised sequence must look exactly like:
        #     SOS, <one or more content tokens>, EOS, PAD, PAD, ...
        # Every other shape is rejected.
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        seq = seq.flatten()

        # Must be non-empty and fully denoised: no MASK tokens may remain.
        if seq.numel() == 0:
            return False
        if (seq == MASK_token).any():
            return False

        # Exactly one SOS, and it must be the very first token.
        if (seq == SOS_token).long().sum() != 1 or seq[0] != SOS_token:
            return False

        # Exactly one EOS, and it must come strictly after the SOS.
        if (seq == EOS_token).long().sum() != 1:
            return False
        eos_position = (seq == EOS_token).nonzero(as_tuple=True)[0].item()
        # eos at index 0 is impossible (SOS is there); index 1 means there is no
        # content between SOS and EOS, which is not a valid sequence.
        if eos_position < 2:
            return False

        # The content strictly between SOS and EOS must be real content tokens
        # only: no PAD, MASK, SOS or EOS may appear inside this region.
        content = seq[1:eos_position]
        special = torch.tensor([SOS_token, EOS_token, PAD_token, MASK_token],
                               device=content.device)
        if torch.isin(content, special).any():
            return False

        # Everything after the EOS must be PAD (only trailing padding allowed).
        if (seq[eos_position + 1:] != PAD_token).any():
            return False

        return True

    def evaluate(self, seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        # Rule checks run on the content up to (and including) the first EOS, so tokens
        # emitted after a valid EOS do not leak into the per-rule counts. Format is still
        # checked on the FULL sequence, so trailing non-PAD garbage is rejected there —
        # this keeps `grammatical` identical while making mean_rule1/mean_rule2 meaningful.
        eos_positions = (seq == EOS_token).nonzero(as_tuple=True)[0]
        eval_seq = seq[:eos_positions[0] + 1] if len(eos_positions) > 0 else seq
        r1 = self.does_satisfy_rule1(eval_seq)
        r2 = self.does_satisfy_rule2(eval_seq)
        fmt = self.does_satisfy_format(seq)
        grammatical = r1 and r2 and fmt # Format matters! Otherwise, e.g., 3 0 4 4 4 1 2 is accepted for aNbN
        return np.array([int(r1), int(r2), int(grammatical), int(fmt)])

    def generate_seq(self):
        """
        Generate all/many valid sequences (already in project token scheme),
        pad to length l+2, and store as a (N, l+2) tensor in self.data.
        """
        re_seqs = _generate_re_sequences(self.grammar_name, self.l)
        target_len = self.l + 2

        rows = []
        for seq_np in re_seqs:
            seq = seq_np.astype(int)
            if len(seq) < target_len:
                seq = np.concatenate([seq, np.full(target_len - len(seq), PAD_token)])
            else:
                seq = seq[:target_len]
            rows.append(seq)

        self.data = torch.tensor(np.stack(rows), dtype=torch.long)
        print(f'RE grammar "{self.grammar_name}" data generated; shape: {self.data.shape}')

    # ------------------------------------------------------------------ #
    # Diversity metric interface                                           #
    # ------------------------------------------------------------------ #

    def valid_n_range(self) -> range:
        """Valid n values for n-distribution grammars.

        aNbN:   n in [1, l//2]   (SOS + n a's + n b's + EOS <= l+2)
        aNbNcN: n in [1, l//3]   (SOS + n a's + n b's + n c's + EOS <= l+2)

        n=0 is excluded because the data generators in re_data start at n=1
        (see generate_aNbN_grammar_data and generate_aNbNcN_grammar_data:
        `lengths = np.linspace(start=1, ...)`).
        """
        if self.grammar_name == 'aNbN':
            return range(1, self.l // 2 + 1)
        elif self.grammar_name == 'aNbNcN':
            return range(1, self.l // 3 + 1)
        raise ValueError(
            f"valid_n_range() not applicable for grammar {self.grammar_name!r}"
        )

    def bbaN_valid_nm_pairs(self) -> frozenset:
        """Valid (n, m) pairs for bbaN per the canonical rule check.

        Per re_data.check_bs_before_as, a sequence in L2 must contain at least
        one b — so n >= 1. Per re_data.check_even_number_of_as_end with m as
        half the trailing a-count, m >= 0 is allowed (pure b^n with no a's is
        valid, corresponding to (n, 0)).

        Length constraint: SOS + n b's + 2m a's + EOS <= total length (l+2),
        so n + 2m <= l.
        """
        pairs = set()
        for n in range(1, self.l + 1):                  # n >= 1 (canonical)
            for m in range(0, (self.l - n) // 2 + 1):   # m >= 0 (FIX: was 1)
                if n + 2 * m <= self.l:
                    pairs.add((n, m))
        return frozenset(pairs)

    def vocab_info(self) -> dict:
        """Token ID map and grammar-specific ranges for diversity_metrics.

        Returns a dict with at minimum: sos, eos, pad, mask.
        Grammar-specific additions:
          baN, bbaN, aNbN, aNbNcN: a, b (and c for aNbNcN)
          bbaN: valid_nm_pairs (frozenset of (n, m) tuples)
          aNbN, aNbNcN: valid_n_range (range)
          parentheses_and_brackets,
          not_nested_parentheses_and_brackets: paren_open, paren_close,
                                               bracket_open, bracket_close
        """
        base = {
            'sos': SOS_token,
            'eos': EOS_token,
            'pad': PAD_token,
            'mask': MASK_token,
        }
        name = self.grammar_name

        if name == 'baN':
            return {**base, 'a': 0, 'b': 1}

        elif name == 'bbaN':
            return {
                **base,
                'a': 0,
                'b': 1,
                'valid_nm_pairs': self.bbaN_valid_nm_pairs(),
            }

        elif name == 'aNbN':
            return {
                **base,
                'a': 0,
                'b': 1,
                'valid_n_range': self.valid_n_range(),
            }

        elif name == 'aNbNcN':
            return {
                **base,
                'a': 0,
                'b': 1,
                'c': 6,
                'valid_n_range': self.valid_n_range(),
            }

        elif name in ('parentheses_and_brackets',
                      'not_nested_parentheses_and_brackets'):
            return {
                **base,
                'paren_open': 0,
                'paren_close': 1,
                'bracket_open': 6,
                'bracket_close': 7,
            }

        else:
            # Fallback for other supported grammars
            return base

    def get_some_known_valid_sequences(self, n: int = 10):
        """Return up to n valid sequences from self.data as a list of tensors."""
        if self.data is None:
            raise RuntimeError("Call generate_seq() first.")
        end = min(n, len(self.data))
        return [self.data[i] for i in range(end)]

    def diversity_metrics(self, correct_sequences) -> dict:
        """Scalar diversity metrics. Returns dict[str, float]; NaN for inapplicable."""
        from diversity_metrics import compute_diversity_metrics
        return compute_diversity_metrics(
            self.grammar_name, correct_sequences,
            vocab=self.vocab_info(), L=self.l,
        )

    def diversity_distributions(self, correct_sequences) -> dict:
        """Raw distributions for JSON sidefile. Returns dict[str, list]."""
        from diversity_metrics import compute_diversity_distributions
        return compute_diversity_distributions(
            self.grammar_name, correct_sequences, vocab=self.vocab_info(),
        )