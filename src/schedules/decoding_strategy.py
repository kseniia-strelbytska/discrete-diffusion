from abc import ABC, abstractmethod
import torch
from torch import Tensor

class DecodingStrategy(ABC):
    @abstractmethod
    def select_positions(self, *, X, content_probs, mask_prob, masked_mask, step, num_steps, device) -> Tensor:
        """
        Returns a boolean mask of length L: True at positions to unmask THIS step.
        Only positions where masked_mask is True may be selected (never re-unmask
        an already-revealed position, never touch SOS at position 0).

        Args:
          X            : (L,) current sequence (mask tokens at unrevealed positions)
          content_probs: (L, V-1) per-position probability over content tokens
                         (NOT including MASK). Already temperature-free probabilities.
          mask_prob    : (L,) schedule-derived probability of STAYING masked this step.
                         Only meaningful for schedule-based strategies; EB ignores it.
          masked_mask  : (L,) bool, True where X == MASK_token.
          step         : int, current denoising step index.
          num_steps    : int, total steps.
        """
        ...


class ScheduleDrivenDecoding(DecodingStrategy):
    """
    Stochastic Bernoulli position selection driven by the noise schedule.

    Works identically for both uniform (CategoricalSchedule) and Gaussian schedules —
    the schedule object determines mask_prob; this class just draws from it.
    """

    def select_positions(self, *, X, content_probs, mask_prob, masked_mask, step, num_steps, device) -> Tensor:
        L = X.shape[0]
        return (torch.rand(L, device=device) < (1 - mask_prob)) & masked_mask


class EBSamplerDecoding(DecodingStrategy):
    """
    Entropy-Bounded position selection (https://arxiv.org/pdf/2505.24857).

    Selects k positions per step adaptively: positions with lowest Shannon
    entropy (most confident predictions) are unmasked first, with k
    determined by a cumulative entropy bound parametrised by gamma.

    Ignores mask_prob, step, num_steps, and the noise schedule entirely.
    """

    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    def select_positions(self, *, X, content_probs, mask_prob, masked_mask, step, num_steps, device) -> Tensor:
        L = X.shape[0]

        # 1. Shannon entropy at each position (natural log).
        eps = 1e-12
        entropy = -(content_probs * (content_probs + eps).log()).sum(dim=-1)  # (L,)

        # 2. Revealed positions get +inf so they sort last and are never selected.
        err = torch.where(masked_mask, entropy, torch.full_like(entropy, float('inf')))

        # 3. Sort masked positions by ascending error (lowest entropy = most confident first).
        order = torch.argsort(err)
        masked_sorted = order[masked_mask[order]]  # indices of masked positions, low-entropy first

        if masked_sorted.numel() == 0:
            return torch.zeros(L, dtype=torch.bool, device=device)

        # 4. Entropy-bounded cumulative criterion.
        ent_sorted = entropy[masked_sorted]                     # (M,)
        acc_entropy = torch.cumsum(ent_sorted, dim=0)           # (M,)
        cummax_ent = torch.cummax(ent_sorted, dim=0).values     # (M,)
        k = int((acc_entropy - cummax_ent <= self.gamma).sum().item())
        k = max(k, 1)
        k = min(k, masked_sorted.numel())
        
        # 5. Boolean mask of selected positions.
        sel = torch.zeros(L, dtype=torch.bool, device=device)
        sel[masked_sorted[:k]] = True
        
        selected_tokens = X[sel]
        selected_entropy = [round(i) for i in entropy[sel].tolist()]
        return sel
