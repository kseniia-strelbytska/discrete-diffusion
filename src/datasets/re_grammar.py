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
        self._grammar_rule = rd.grammar_rules(grammar_name)

    # ------------------------------------------------------------------ #
    # FormalGrammar interface                                              #
    # ------------------------------------------------------------------ #

    def does_satisfy_rule1(self, seq):
        """Grammaticality: satisfies the full RE grammar rule."""
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        try:
            return bool(self._grammar_rule(seq))
        except Exception:
            return False

    def does_satisfy_rule2(self, seq):
        """Always True — RE grammars expose a single combined rule."""
        return True

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
        grammatical = self.does_satisfy_rule1(seq)
        fmt = self.does_satisfy_format(seq)
        return np.array([int(grammatical), 1, int(grammatical), int(fmt)])

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
