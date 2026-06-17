"""
Tests for the decoupled DecodingStrategy / SamplingStrategy refactor.

Covers:
  1. Equivalence: schedule-driven + categorical matches the old joint-multinomial distribution.
  2. EBSamplerDecoding: k (positions unmasked) increases monotonically with gamma.
  3. GreedySampling is deterministic; CategoricalSampling varies across runs.
  4. DecodingStrategy contract: never selects position 0 (SOS) or already-revealed positions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
import torch
from datasets.constants import MASK_token, SOS_token
from schedules.decoding_strategy import ScheduleDrivenDecoding, EBSamplerDecoding
from schedules.sampling_strategy import GreedySampling, CategoricalSampling
from schedules.categorical_schedule import CategoricalSchedule


VOCAB_SIZE = 6  # tokens 0..4 are content, 5 = MASK_token


# ─── helpers ─────────────────────────────────────────────────────────────────

def _content_idx(vocab_size=VOCAB_SIZE):
    return torch.tensor([j for j in range(vocab_size) if j != MASK_token], dtype=torch.long)


def _random_content_probs(L, vocab_size=VOCAB_SIZE, seed=0):
    torch.manual_seed(seed)
    idx = _content_idx(vocab_size)
    return torch.softmax(torch.randn(L, len(idx)), dim=-1)


def _make_schedule_strategy():
    return ScheduleDrivenDecoding(CategoricalSchedule())


# ─── Test 1: Equivalence ─────────────────────────────────────────────────────

def _old_step_unmasked(content_probs, content_idx, mask_prob, weight):
    """Reference: old joint-multinomial logic returns bool mask of unmasked positions."""
    L = content_probs.shape[0]
    V = content_probs.shape[-1] + 1  # +1 for MASK slot

    probs = torch.zeros(L, V)
    probs[:, content_idx] = content_probs * weight
    probs[:, MASK_token] = mask_prob
    probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    zero_rows = probs.sum(dim=-1) == 0
    probs[zero_rows, MASK_token] = 1.0

    sampled = torch.multinomial(probs, 1).squeeze(-1)
    return sampled != MASK_token


def test_equivalence_schedule_categorical():
    """
    ScheduleDrivenDecoding + CategoricalSampling produces the same per-position
    unmasking probability as the old joint multinomial over [content * weight, mask_prob].

    Both have P(unmask at pos i) = 1 - mask_prob[i]; the factorisation is exact
    because weight = 1 - mask_prob and content choice is independent of the
    unmask/stay-masked decision.
    """
    NUM_TRIALS = 800
    L = 12
    timestep = torch.tensor(0.5)
    dt = 0.1

    schedule = CategoricalSchedule()

    # Derive the expected mask_prob from the schedule (same math as ScheduleDrivenDecoding).
    alpha_t = 1 - schedule.p_mask(timestep, max_l=L, device='cpu')
    alpha_s = 1 - schedule.p_mask(timestep - dt, max_l=L, device='cpu')
    weight = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
    mask_prob = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0)
    mask_prob_1d = mask_prob.squeeze(0)           # (L,)
    weight_col   = weight.squeeze(0).unsqueeze(-1)  # (L, 1)

    content_probs = _random_content_probs(L, seed=42)
    content_idx = _content_idx()
    masked_mask = torch.ones(L, dtype=torch.bool)
    X = torch.full((L,), MASK_token, dtype=torch.long)

    # Old approach: joint multinomial.
    torch.manual_seed(0)
    old_counts = torch.zeros(L)
    for _ in range(NUM_TRIALS):
        old_counts += _old_step_unmasked(content_probs, content_idx, mask_prob_1d, weight_col).float()

    # New approach: ScheduleDrivenDecoding (owns the alpha / mask_prob math internally).
    torch.manual_seed(0)
    new_counts = torch.zeros(L)
    decoding = ScheduleDrivenDecoding(schedule)
    for _ in range(NUM_TRIALS):
        sel = decoding.select_positions(
            X=X, content_probs=content_probs, masked_mask=masked_mask,
            timestep=timestep, dt=dt, device='cpu',
        )
        new_counts += sel.float()

    old_rates = old_counts / NUM_TRIALS
    new_rates = new_counts / NUM_TRIALS
    # Tolerance accounts for sampling noise (≈2–3 sigma at NUM_TRIALS=800).
    assert torch.allclose(old_rates, new_rates, atol=0.09), (
        f"Unmasking rates diverge between old and new: "
        f"old_mean={old_rates.mean():.3f}, new_mean={new_rates.mean():.3f}"
    )


# ─── Test 2: EB k increases with gamma ───────────────────────────────────────

def test_eb_k_increases_with_gamma():
    """EBSamplerDecoding with larger gamma selects more positions per step."""
    torch.manual_seed(1)
    L = 20
    X = torch.full((L,), MASK_token, dtype=torch.long)
    X[0] = SOS_token
    masked_mask = (X == MASK_token)

    content_probs = _random_content_probs(L, seed=1)

    gammas = [0.005, 0.05, 0.5, 2.0, 10.0]
    k_values = []
    for gamma in gammas:
        eb = EBSamplerDecoding(gamma=gamma)
        sel = eb.select_positions(
            X=X, content_probs=content_probs, masked_mask=masked_mask, device='cpu',
        )
        k_values.append(int(sel.sum().item()))

    for i in range(len(k_values) - 1):
        assert k_values[i] <= k_values[i + 1], (
            f"k not monotone with gamma at index {i}: gammas={gammas}, k_values={k_values}"
        )


def test_eb_always_unmasks_at_least_one():
    """EBSamplerDecoding selects at least one position even when gamma is tiny."""
    torch.manual_seed(2)
    L = 10
    X = torch.full((L,), MASK_token, dtype=torch.long)
    X[0] = SOS_token
    masked_mask = (X == MASK_token)

    content_probs = _random_content_probs(L, seed=2)

    eb = EBSamplerDecoding(gamma=0.0)
    sel = eb.select_positions(
        X=X, content_probs=content_probs, masked_mask=masked_mask, device='cpu',
    )
    assert sel.sum().item() >= 1


# ─── Test 3: Greedy is deterministic; categorical varies ─────────────────────

def test_greedy_is_deterministic():
    """GreedySampling produces identical output on every call for fixed input."""
    torch.manual_seed(3)
    L = 10
    content_idx = _content_idx()
    content_probs = _random_content_probs(L, seed=3)
    positions_mask = torch.ones(L, dtype=torch.bool)

    greedy = GreedySampling()
    first = greedy.choose_tokens(
        content_probs=content_probs, content_idx=content_idx,
        positions_mask=positions_mask, device='cpu',
    )
    for _ in range(5):
        result = greedy.choose_tokens(
            content_probs=content_probs, content_idx=content_idx,
            positions_mask=positions_mask, device='cpu',
        )
        assert torch.equal(first, result), "GreedySampling must be deterministic"


def test_categorical_varies_across_runs():
    """CategoricalSampling produces different outputs across calls (statistically)."""
    torch.manual_seed(4)
    L = 20
    content_idx = _content_idx()
    content_probs = torch.full((L, len(content_idx)), 1.0 / len(content_idx))
    positions_mask = torch.ones(L, dtype=torch.bool)

    cat = CategoricalSampling()
    results = [
        cat.choose_tokens(
            content_probs=content_probs, content_idx=content_idx,
            positions_mask=positions_mask, device='cpu',
        )
        for _ in range(10)
    ]
    any_different = any(not torch.equal(results[0], r) for r in results[1:])
    assert any_different, "CategoricalSampling should vary across runs"


# ─── Test 4: DecodingStrategy contract ───────────────────────────────────────

def _assert_decoding_contract(strategy, X, content_probs, masked_mask, label,
                              *, timestep=None, dt=None):
    if isinstance(strategy, ScheduleDrivenDecoding):
        sel = strategy.select_positions(
            X=X, content_probs=content_probs, masked_mask=masked_mask,
            timestep=timestep, dt=dt, device='cpu',
        )
    else:
        sel = strategy.select_positions(
            X=X, content_probs=content_probs, masked_mask=masked_mask, device='cpu',
        )
    assert not sel[0].item(), f"{label}: selected position 0 (SOS)"
    revealed = ~masked_mask
    assert not (sel & revealed).any().item(), (
        f"{label}: selected an already-revealed position"
    )


def test_decoding_never_selects_sos_or_revealed():
    """Both DecodingStrategies must never touch SOS (pos 0) or revealed positions."""
    torch.manual_seed(5)
    L = 14
    X = torch.full((L,), MASK_token, dtype=torch.long)
    X[0] = SOS_token
    X[3] = 1
    X[8] = 2
    masked_mask = (X == MASK_token)

    content_probs = _random_content_probs(L, seed=5)
    timestep = torch.tensor(0.5)
    dt = 0.1

    strategies = [
        (_make_schedule_strategy(), "ScheduleDrivenDecoding"),
        (EBSamplerDecoding(gamma=100.0), "EBSamplerDecoding(gamma=100)"),
    ]

    for _ in range(30):
        for strat, label in strategies:
            _assert_decoding_contract(
                strat, X, content_probs, masked_mask, label,
                timestep=timestep, dt=dt,
            )


def test_decoding_selects_only_masked_positions():
    """select_positions must return True only where masked_mask is True."""
    torch.manual_seed(6)
    L = 10
    X = torch.full((L,), MASK_token, dtype=torch.long)
    X[0] = SOS_token
    X[2] = 0
    X[5] = 1
    masked_mask = (X == MASK_token)

    content_probs = _random_content_probs(L, seed=6)
    timestep = torch.tensor(0.5)
    dt = 0.1

    strategies = [_make_schedule_strategy(), EBSamplerDecoding(gamma=5.0)]
    for strat in strategies:
        for _ in range(20):
            if isinstance(strat, ScheduleDrivenDecoding):
                sel = strat.select_positions(
                    X=X, content_probs=content_probs, masked_mask=masked_mask,
                    timestep=timestep, dt=dt, device='cpu',
                )
            else:
                sel = strat.select_positions(
                    X=X, content_probs=content_probs, masked_mask=masked_mask, device='cpu',
                )
            assert not (sel & ~masked_mask).any().item(), (
                f"{strat.__class__.__name__} selected a non-masked position"
            )


# ─── Test 5: SamplingStrategy fills non-selected positions with MASK_token ───

def test_sampling_fills_non_selected_with_mask():
    """choose_tokens must return MASK_token at positions where positions_mask is False."""
    torch.manual_seed(7)
    L = 8
    content_idx = _content_idx()
    content_probs = _random_content_probs(L, seed=7)
    positions_mask = torch.tensor([True, False, True, False, True, False, True, False])

    for strat in [GreedySampling(), CategoricalSampling()]:
        tokens = strat.choose_tokens(
            content_probs=content_probs, content_idx=content_idx,
            positions_mask=positions_mask, device='cpu',
        )
        assert (tokens[~positions_mask] == MASK_token).all(), (
            f"{strat.__class__.__name__} did not fill non-selected positions with MASK_token"
        )
        assert (tokens[positions_mask] != MASK_token).all(), (
            f"{strat.__class__.__name__} returned MASK_token at a selected position"
        )
