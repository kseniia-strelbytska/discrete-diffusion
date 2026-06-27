"""Unmasking schedules and samplers, ported from ``src/schedules/``.

Everything here operates on a *canvas* of length ``M`` (the generated region)
with an explicit ``mask_id`` and full-vocabulary probabilities, so the same code
drives a real BPE-vocab diffusion model instead of the 6-8 token grammar oracle.

Position selectors return a boolean tensor of length ``M`` — True where a canvas
slot should be unmasked THIS step. Only currently-masked slots may be selected.

Sources (formulas copied 1:1):
  - Gaussian p_mask                : src/schedules/gaussian_schedule.py:36-53
  - schedule-driven Bernoulli draw : src/schedules/decoding_strategy.py:62-70
  - entropy-bounded (EB) selection : src/schedules/decoding_strategy.py:116-158
  - autoregressive selection       : src/schedules/decoding_strategy.py:86-95
  - greedy / categorical sampling  : src/schedules/sampling_strategy.py:23-45
"""

from __future__ import annotations

import torch
from torch import Tensor


# ── noise-schedule p_mask profiles ──────────────────────────────────────────

def uniform_p_mask(t: float, M: int, device) -> Tensor:
    """Position-independent MDLM schedule: p_mask(t) = t for every position."""
    return torch.full((M,), float(t), device=device)


def gaussian_p_mask(t: float, M: int, sigma: float, device) -> Tensor:
    """Position-dependent Gaussian-CDF schedule (left-to-right reveal bias).

    Mean sweeps right (t=0, p_mask~0) to left (t=1, p_mask~1); sigma sets the
    sharpness of the reveal front. Ported from GaussianSchedule.get_dist/p_mask.
    """
    x = torch.arange(M, dtype=torch.float32, device=device)
    margin = 2.0 * sigma
    width = M + 2.0 * margin
    mean = (1.0 - t) * width - margin
    dist = torch.distributions.Normal(mean, sigma)
    return dist.cdf(x)


def schedule_unmask_prob(p_mask_t: Tensor, p_mask_s: Tensor) -> Tensor:
    """Per-position probability of unmasking when moving t -> s (s < t).

    Identical algebra to ScheduleDrivenDecoding.select_positions:
        alpha_t = 1 - p_mask(t);  alpha_s = 1 - p_mask(s)
        mask_prob = (1 - alpha_s) / (1 - alpha_t)
        unmask_prob = 1 - mask_prob
    """
    alpha_t = 1.0 - p_mask_t
    alpha_s = 1.0 - p_mask_s
    mask_prob = ((1.0 - alpha_s) / (1.0 - alpha_t).clamp(min=1e-12)).clamp(0.0, 1.0)
    return (1.0 - mask_prob).clamp(0.0, 1.0)


# ── position selectors (one canvas row, length M) ────────────────────────────

def select_schedule(unmask_prob: Tensor, masked_mask: Tensor) -> Tensor:
    """Bernoulli draw on currently-masked slots (uniform & Gaussian)."""
    draw = torch.rand(masked_mask.shape[0], device=masked_mask.device) < unmask_prob
    return draw & masked_mask


def select_autoregressive(masked_mask: Tensor) -> Tensor:
    """Unmask exactly the leftmost still-masked slot."""
    sel = torch.zeros_like(masked_mask)
    idx = masked_mask.nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        sel[idx[0]] = True
    return sel


def select_eb(content_probs: Tensor, masked_mask: Tensor, gamma: float) -> Tensor:
    """Entropy-bounded adaptive selection.

    content_probs : (M, V) probabilities over content tokens (mask excluded/zeroed).
    Selects the lowest-entropy masked slots whose cumulative (entropy - running max)
    stays within gamma; always at least one. Ported from EBSamplerDecoding.
    """
    eps = 1e-12
    entropy = -(content_probs * (content_probs + eps).log()).sum(dim=-1)  # (M,)
    err = torch.where(masked_mask, entropy, torch.full_like(entropy, float("inf")))
    order = torch.argsort(err)
    masked_sorted = order[masked_mask[order]]

    sel = torch.zeros_like(masked_mask)
    if masked_sorted.numel() == 0:
        return sel

    ent_sorted = entropy[masked_sorted]
    acc = torch.cumsum(ent_sorted, dim=0)
    cummax = torch.cummax(ent_sorted, dim=0).values
    k = int((acc - cummax <= gamma).sum().item())
    k = max(1, min(k, masked_sorted.numel()))
    sel[masked_sorted[:k]] = True
    return sel


# ── token samplers ───────────────────────────────────────────────────────────

def choose_greedy(content_probs: Tensor) -> Tensor:
    """Argmax content token per slot. content_probs: (M, V) -> (M,) vocab ids."""
    return content_probs.argmax(dim=-1)


def choose_categorical(content_probs: Tensor) -> Tensor:
    """Multinomial draw per slot. content_probs: (M, V) -> (M,) vocab ids."""
    row_sums = content_probs.sum(dim=-1, keepdim=True)
    safe = content_probs.clone()
    zero = (row_sums.squeeze(-1) == 0)
    if zero.any():
        safe[zero] = 1.0 / content_probs.shape[-1]
    return torch.multinomial(safe, 1).squeeze(-1)


def content_probs_from_logits(logits: Tensor, mask_id: int, temperature: float) -> Tensor:
    """Softmax over the vocab with the mask column removed (set to 0 after softmax).

    logits : (M, V). Returns (M, V) probs with column ``mask_id`` zeroed and rows
    renormalised, so the mask token can never be sampled and EB entropy is taken
    over real tokens only. Mirrors _content_probs (logit branch) + temperature.
    """
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    probs[:, mask_id] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return probs
