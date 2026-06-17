from abc import ABC, abstractmethod
import torch
from torch import Tensor


class DecodingStrategy(ABC):
    """
    Base class for position-selection strategies used by ScheduledUnmasker.

    Two concrete variants exist:
      - ScheduleDrivenDecoding  (needs_schedule = True)  — Bernoulli draws driven by
        the noise-schedule's mask_prob. Owns all alpha/weight/mask_prob mathematics.
      - EBSamplerDecoding       (needs_schedule = False) — adaptive entropy-bounded
        selection. Ignores the noise schedule and timestep entirely.

    The outer loop in ScheduledUnmasker.forward() branches on needs_schedule to
    provide each strategy with exactly the arguments it requires.
    """

    needs_schedule: bool = False

    @abstractmethod
    def select_positions(self, *, X, content_probs, masked_mask, device, **kwargs) -> Tensor:
        """
        Returns a boolean mask of length L: True at positions to unmask THIS step.
        Only positions where masked_mask is True may be selected.
        """
        ...


class ScheduleDrivenDecoding(DecodingStrategy):
    """
    Stochastic Bernoulli position selection driven by the noise schedule.

    Owns all schedule-dependent calculations: alpha_t, alpha_s, weight, mask_prob.
    Works identically for CategoricalSchedule and GaussianSchedule — the schedule
    object controls the mask_prob profile; this class draws from it.

    After each call to select_positions(), last_weight (L, 1) and last_mask_prob (L,)
    are stored so the caller can build diagnostic probability tensors for error tracking.
    """

    needs_schedule = True

    def __init__(self, schedule):
        self.schedule = schedule
        self.last_weight: Tensor | None = None
        self.last_mask_prob: Tensor | None = None

    def select_positions(self, *, X, content_probs, masked_mask,
                         timestep, dt, device) -> Tensor:
        """
        Args:
          X           : (L,) current sequence
          content_probs: (L, V-1) — unused here; present for interface symmetry
          masked_mask : (L,) bool, True where X == MASK_token
          timestep    : scalar tensor, current denoising time t
          dt          : float, step size Δt
          device      : torch.device
        """
        L = X.shape[0]
        alpha_t = 1 - self.schedule.p_mask(timestep, max_l=L, device=device)
        alpha_s = 1 - self.schedule.p_mask(timestep - dt, max_l=L, device=device)

        weight = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
        mask_prob = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0)
        self.last_weight = weight.squeeze(0).unsqueeze(-1)  # (L, 1)
        self.last_mask_prob = mask_prob.squeeze(0)           # (L,)

        return (torch.rand(L, device=device) < (1 - self.last_mask_prob)) & masked_mask


class AutoregressiveDecoding(DecodingStrategy):
    """
    Autoregressive position selection: unmask exactly the first (leftmost)
    masked token on every timestep.

    needs_schedule = False — no noise-schedule math is required.  The outer
    loop in ScheduledUnmasker.forward() automatically caps max_steps to
    X.shape[0] when this strategy is active, guaranteeing that the number of
    denoising steps equals the sequence length.
    """

    needs_schedule = False

    def select_positions(self, *, X, content_probs=None, masked_mask, device) -> Tensor:
        """
        Returns a boolean mask with True only at the lowest-index masked position.
        If no masked positions remain, returns an all-False mask.
        """
        sel = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
        masked_positions = masked_mask.nonzero(as_tuple=True)[0]
        if masked_positions.numel() > 0:
            sel[masked_positions[0]] = True
        return sel


class EBSamplerDecoding(DecodingStrategy):
    """
    Entropy-Bounded position selection (https://arxiv.org/pdf/2505.24857).

    Fully adaptive: ignores the noise schedule, timestep, dt, and step count.
    Selects k positions per step based on Shannon entropy (natural log), with k
    bounded by a cumulative entropy threshold parametrised by gamma.

    The loop in ScheduledUnmasker.forward() runs until all masked positions are
    filled or MAX_STEPS is reached — no pre-computed timestep tensor required.
    """

    needs_schedule = False
    MAX_STEPS: int = 10**5

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    def select_positions(self, *, X, content_probs, masked_mask, device) -> Tensor:
        """
        Args:
          X           : (L,) current sequence
          content_probs: (L, V-1) probabilities over content tokens
          masked_mask : (L,) bool, True where X == MASK_token
          device      : torch.device
        """
        L = X.shape[0]

        # Shannon entropy at each position (natural log).
        eps = 1e-12
        entropy = -(content_probs * (content_probs + eps).log()).sum(dim=-1)  # (L,)

        # Revealed positions get +inf so they sort last and are never selected.
        err = torch.where(masked_mask, entropy, torch.full_like(entropy, float('inf')))

        # Sort masked positions by ascending entropy (lowest = most confident first).
        order = torch.argsort(err)
        masked_sorted = order[masked_mask[order]]

        if masked_sorted.numel() == 0:
            return torch.zeros(L, dtype=torch.bool, device=device)

        # Entropy-bounded cumulative criterion.
        ent_sorted = entropy[masked_sorted]
        acc_entropy = torch.cumsum(ent_sorted, dim=0)
        cummax_ent = torch.cummax(ent_sorted, dim=0).values
        k = int((acc_entropy - cummax_ent <= self.gamma).sum().item())
        k = max(k, 1)
        k = min(k, masked_sorted.numel())

        sel = torch.zeros(L, dtype=torch.bool, device=device)
        sel[masked_sorted[:k]] = True
        return sel
