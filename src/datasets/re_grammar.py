## Taken from https://github.com/meszarosanna/rule_extrapolation/tree/main/rule_extrapolation

"""
REGrammar: wraps RE_data generators into the FormalGrammar interface.

re_data.py now uses the project token scheme directly:
  SOS=3, EOS=2, PAD=4, MASK=5
  A=0, B=1, C=6
  open_paren=0, close_paren=1, open_bracket=6, close_bracket=7

No remapping is needed — grammar check functions in re_data.py operate
on project token IDs, and generated sequences are already in project scheme.

Grammar vocab_sizes:
  aNbN, abN, baN, bbaN, aNbM, aNbNaN  → vocab_size=6  (A=0, B=1)
  parentheses                          → vocab_size=6  (open=0, close=1)
  aNbNcN                               → vocab_size=7  (A=0, B=1, C=6)
  brackets                             → vocab_size=8  (open=6, close=7)
  parentheses_and_brackets,
  separated_parentheses_and_brackets,
  not_nested_parentheses_and_brackets  → vocab_size=8  (open_p=0, close_p=1,
                                                         open_b=6, close_b=7)

Rule assignments per language category (table):
  L1 = baN                               regular         rule1=#a even,          rule2=starts with b
  L2 = bbaN                              regular         rule1=#a even,          rule2=b's before a's
  L3 = aNbN                              context-free    rule1=#a=#b,            rule2=a's before b's
  L4 = Dyck (parentheses_and_brackets)   context-free    rule1=paired/nested [],  rule2=paired/nested ()
  L5 = aNbNcN                            context-sensitive rule1=#a=#b=#c,       rule2=a's before b's before c's
  L6 = CS Dyck (not_nested_…)            context-sensitive rule1=paired [],      rule2=paired ()
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

# Maps each grammar to (rule1_fn, rule2_fn).
# rule1 and rule2 are the two independent structural properties.
_GRAMMAR_RULES = {
    # L1 = {bα}: rule1=#a even, rule2=starts with b
    'baN': (rd.check_even_number_of_as, rd.check_begins_with_b),
    # L2 = {b^n a^{2m}}: rule1=#a even, rule2=b's before a's
    'bbaN': (rd.check_even_number_of_as, rd.check_bs_before_as),
    # L3 = {a^n b^n}: rule1=#a=#b, rule2=a's before b's
    'aNbN': (rd.check_same_number_as_bs, rd.check_as_before_bs),
    # L4 = Dyck: rule1=paired and nested [], rule2=paired and nested ()
    'parentheses_and_brackets': (rd.check_matched_brackets, rd.check_matched_parentheses),
    # L5 = {a^n b^n c^n}: rule1=#a=#b=#c, rule2=a's before b's before c's
    'aNbNcN': (rd.check_same_number_as_bs_cs, rd.check_as_before_bs_before_cs),
    # L6 = CS Dyck (not nested): rule1=paired [], rule2=paired ()
    'not_nested_parentheses_and_brackets': (rd.check_matched_brackets, rd.check_matched_parentheses),
    # remaining grammars: use combined grammar rule as rule1, rule2 always True
    'abN': (rd.check_same_number_as_bs, lambda _: True),
    'aNbM': (rd.check_as_before_bs, lambda _: True),
    'aNbNaN': (
        lambda x: rd.check_twice_many_as_than_bs(x) and rd.check_bs_in_the_middle(x) and rd.check_bs_together(x),
        lambda _: True,
    ),
    'parentheses': (rd.check_matched_parentheses, lambda _: True),
    'brackets': (rd.check_matched_brackets, lambda _: True),
    'separated_parentheses_and_brackets': (rd.check_separated_brackets_and_parentheses, lambda _: True),
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
    the language category table in the module docstring.
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
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        if (seq == MASK_token).long().sum() != 0:
            return False
        sos_count = (seq == SOS_token).long().sum()
        eos_count = (seq == EOS_token).long().sum()
        if sos_count != 1 or eos_count != 1:
            return False
        if seq[0] != SOS_token:
            return False
        return True

    def evaluate(self, seq):
        r1 = self.does_satisfy_rule1(seq)
        r2 = self.does_satisfy_rule2(seq)
        grammatical = r1 and r2
        fmt = self.does_satisfy_format(seq)
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
