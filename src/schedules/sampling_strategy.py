from abc import ABC, abstractmethod
import torch
from torch import Tensor
from datasets.constants import MASK_token


class SamplingStrategy(ABC):
    @abstractmethod
    def choose_tokens(self, *, content_probs, content_idx, positions_mask, device) -> Tensor:
        """
        For each position where positions_mask is True, return the chosen content
        token (a real vocab id, never MASK). For positions where positions_mask is
        False, returns MASK_token.

        Args:
          content_probs : (L, V-1) probability over content tokens.
          content_idx   : (V-1,) mapping column index -> real vocab token id.
          positions_mask: (L,) bool, True where a token must be chosen.
        """
        ...


class GreedySampling(SamplingStrategy):
    """Picks the highest-probability content token at each selected position."""

    def choose_tokens(self, *, content_probs, content_idx, positions_mask, device) -> Tensor:
        chosen_col = content_probs.argmax(dim=-1)       # (L,)
        tokens = content_idx[chosen_col]                 # (L,) real vocab ids
        return torch.where(positions_mask, tokens, torch.full_like(tokens, MASK_token))


class CategoricalSampling(SamplingStrategy):
    """Samples a content token at each position from content_probs via multinomial."""

    def choose_tokens(self, *, content_probs, content_idx, positions_mask, device) -> Tensor:
        # Guard against all-zero rows: fall back to uniform for those.
        row_sums = content_probs.sum(dim=-1, keepdim=True)
        safe_probs = content_probs.clone()
        zero_rows = (row_sums.squeeze(-1) == 0)
        if zero_rows.any():
            safe_probs[zero_rows] = 1.0 / content_probs.shape[-1]

        chosen_col = torch.multinomial(safe_probs, 1).squeeze(-1)  # (L,)
        tokens = content_idx[chosen_col]                            # (L,) real vocab ids
        return torch.where(positions_mask, tokens, torch.full_like(tokens, MASK_token))
