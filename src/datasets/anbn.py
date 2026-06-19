import torch
import torch.nn as nn
import numpy as np
from .formal_grammar import FormalGrammar
from .constants import EOS_token, SOS_token, PAD_token, MASK_token


class anbnGrammar(FormalGrammar):
    grammar_name = 'aNbN'
    vocab_size = 6

    def __init__(self, l):
        super().__init__(l)
        self.data = None
        self.default_eval_type = 'next_token'

    def does_satisfy_rule1(self, seq):
        zeros = (seq == 0).sum()
        ones = (seq == 1).sum()
        return (zeros == ones)

    def does_satisfy_rule2(self, seq):
        zero, one = False, False
        for idx in range(0, len(seq)):
            if seq[idx] != 0 and seq[idx] != 1:
                continue
            if seq[idx] == 0:
                zero = True
                if one == True:
                    return False
            else:
                one = True
        return True

    def does_satisfy_format(self, seq):
        if (seq == MASK_token).long().sum() != 0:
            return False

        SOS_token_count, EOS_token_count = (seq == SOS_token).long().sum(), (seq == EOS_token).long().sum()
        if SOS_token_count != 1 or EOS_token_count != 1:
            return False

        if seq[0] != SOS_token or seq[-1] != EOS_token:
            return False

        zero_count, one_count = (seq[1:-1] == 0).long().sum(), (seq[1:-1] == 1).long().sum()
        if zero_count + one_count != len(seq) - 2:
            return False

        return True

    def evaluate(self, seq):
        eos_positions = torch.where(seq == EOS_token)[0]
        eval_seq = seq[:eos_positions[0] + 1] if len(eos_positions) > 0 else seq

        a = self.does_satisfy_rule1(eval_seq)
        b = self.does_satisfy_rule2(eval_seq)
        c = self.does_satisfy_format(eval_seq)

        return np.array([int(i) for i in [a, b, (a & b), c]])

    def generate_seq(self):
        data = None

        for l in range(2, self.l + 1, 2):
            seq = torch.cat([torch.tensor([SOS_token]),
                            torch.zeros(l // 2),
                            torch.ones(l // 2),
                            torch.tensor([EOS_token]),
                            torch.full((self.l - l,), torch.tensor(PAD_token))], dim=-1).long().unsqueeze(0)

            if data is None:
                data = seq.clone()
            else:
                data = torch.cat([data, seq], dim=0)

        print(f'Data generated; shape: {data.shape}')
        self.data = data

    def valid_n_range(self) -> range:
        """Valid n values for a^n b^n sequences: n in [1, l//2]."""
        return range(1, self.l // 2 + 1)

    def vocab_info(self) -> dict:
        """Token ID map and grammar-specific ranges for diversity_metrics."""
        return {
            'sos': SOS_token,
            'eos': EOS_token,
            'pad': PAD_token,
            'mask': MASK_token,
            'a': 0,
            'b': 1,
            'valid_n_range': self.valid_n_range(),
        }

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
