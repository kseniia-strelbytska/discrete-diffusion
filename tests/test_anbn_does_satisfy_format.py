import os
import sys

import pytest
import torch

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from anbn import anbnGrammar
from constants import EOS_token, MASK_token, PAD_token, SOS_token


@pytest.mark.parametrize(
    "seq,expected",
    [
        # SOS, 1, 0, 1, EOS, PAD, PAD -> [rule1, rule2, both, format]
        (
            [SOS_token, 1, 0, 1, EOS_token, PAD_token, PAD_token],
            [0, 0, 0, 1],
        ),
        # EOS, 0, 1, SOS, 0 -> missing SOS before EOS in the evaluated prefix [EOS]
        (
            [EOS_token, 0, 1, SOS_token, 0],
            [1, 1, 1, 0],
        ),
        # SOS, 0, 0, PAD, 1, 0, EOS, 0, 1, PAD, PAD -> PAD among numbers before EOS
        (
            [SOS_token, 0, 0, PAD_token, 1, 0, EOS_token, 0, 1, PAD_token, PAD_token],
            [0, 0, 0, 0],
        ),
        # 0, 0, 1, 1, EOS -> missing SOS
        (
            [0, 0, 1, 1, EOS_token],
            [1, 1, 1, 0],
        ),
        # PAD, SOS, 0, 1, EOS -> SOS not first
        (
            [PAD_token, SOS_token, 0, 1, EOS_token],
            [1, 1, 1, 0],
        ),
        # SOS, EOS, 0, 1, PAD -> first EOS truncation leaves minimal valid sequence
        (
            [SOS_token, EOS_token, 0, 1, PAD_token],
            [1, 1, 1, 1],
        ),
        # SOS, 0, EOS, 1, EOS -> first EOS defines evaluated prefix
        (
            [SOS_token, 0, EOS_token, 1, EOS_token],
            [0, 1, 0, 1],
        ),
        # SOS, 0, MASK, 1, EOS -> MASK inside evaluated region should fail format
        (
            [SOS_token, 0, MASK_token, 1, EOS_token],
            [1, 1, 1, 0],
        ),
        # SOS, 0, SOS, 1, EOS -> duplicate SOS token should fail format
        (
            [SOS_token, 0, SOS_token, 1, EOS_token],
            [1, 1, 1, 0],
        ),
        # SOS, 1, 1, 0, 0, EOS -> format valid but rule2 violated
        (
            [SOS_token, 1, 1, 0, 0, EOS_token],
            [1, 0, 0, 1],
        ),
    ],
)
def test_does_satisfy_format_sequences(seq, expected):
    g = anbnGrammar(l=len(seq))
    stats = g.evaluate(torch.tensor(seq).long())

    assert stats.tolist() == expected
