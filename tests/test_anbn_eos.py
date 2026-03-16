import os
import sys
import torch

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from anbn import anbnGrammar
from constants import SOS_token, EOS_token, PAD_token, MASK_token


def test_eos_ignores_tokens_for_rule1():
    # Sequence where tokens after EOS would change rule1 if considered
    # SOS, 0, 1, EOS, 0, PAD
    seq = torch.tensor([SOS_token, 0, 1, EOS_token, 0, PAD_token]).long()

    g = anbnGrammar(l=6)

    # replicate eval_seq logic from the grammar to show what is used for evaluation
    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for rule1 test:", eval_seq.tolist())

    stats = g.evaluate(seq)

    # Rule1 should be True when only the prefix up to EOS is considered
    assert stats[0] == 1


def test_eos_ignores_tokens_for_format():
    # Sequence where a token after EOS would break the format check
    # SOS, 0, EOS, 0, PAD -> format should be valid when only prefix is considered
    seq = torch.tensor([SOS_token, 0, EOS_token, 0, PAD_token]).long()

    g = anbnGrammar(l=5)

    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for format test:", eval_seq.tolist())

    stats = g.evaluate(seq)

    # Format check (last element of stats) should be True
    assert stats[3] == 1


def test_no_eos_behavior():
    # Sequence without EOS: full sequence is used for evaluation
    # SOS, 0, 0, 1, PAD -> zeros=2, ones=1 => rule1 False; format False (EOS missing)
    seq = torch.tensor([SOS_token, 0, 0, 1, PAD_token]).long()

    g = anbnGrammar(l=5)

    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for no_eos:", eval_seq.tolist())

    stats = g.evaluate(seq)

    assert stats[0] == 0
    assert stats[3] == 0


def test_multiple_eos_uses_first():
    # Ensure only the first EOS is considered
    seq = torch.tensor([SOS_token, 0, 1, EOS_token, 0, EOS_token, PAD_token]).long()

    g = anbnGrammar(l=7)

    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for multiple_eos:", eval_seq.tolist())

    stats = g.evaluate(seq)

    assert stats[0] == 1
    assert stats[3] == 1


def test_mask_after_eos_ignored():
    # MASK after EOS should be ignored for format check
    seq = torch.tensor([SOS_token, 0, EOS_token, MASK_token, PAD_token]).long()

    g = anbnGrammar(l=5)

    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for mask_after_eos:", eval_seq.tolist())

    stats = g.evaluate(seq)

    assert stats[3] == 1


def test_eos_immediately_after_sos():
    # Minimal valid sequence: SOS, EOS, PAD
    seq = torch.tensor([SOS_token, EOS_token, PAD_token]).long()

    g = anbnGrammar(l=3)

    eos_positions = torch.where(seq == EOS_token)[0]
    eval_seq = seq[: eos_positions[0] + 1] if len(eos_positions) > 0 else seq

    print("eval_seq for sos_eos:", eval_seq.tolist())

    stats = g.evaluate(seq)

    assert stats[0] == 1
    assert stats[1] == 1
    assert stats[3] == 1
