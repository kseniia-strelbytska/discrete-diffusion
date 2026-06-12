from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


class NoiseSchedule(ABC):
    """
    Abstract base class for discrete diffusion noise schedules.

    All concrete schedules must implement p_mask and dp_mask, which define
    the per-position masking probability and its time-derivative for a
    given normalised timestep t ∈ [0, 1].

    Convention
    ----------
    t = 0  →  clean data    (p_mask ≈ 0, nothing masked)
    t = 1  →  fully noisy   (p_mask ≈ 1, everything masked)

    Shape contract
    --------------
    t may be:
      • a Python float or 0-dim tensor  →  returns (1, max_l)
      • a (B, 1) tensor                 →  returns (B, max_l)
    """

    @abstractmethod
    def p_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        """
        Element-wise masking probability at timestep t.

        Returns a tensor with values in [0, 1] that encodes how likely
        each sequence position is to be masked at time t.

        Args:
            t:      Timestep — float, 0-dim tensor, or (B, 1) tensor.
            max_l:  Sequence length (number of positions).
            device: Target device for the returned tensor.

        Returns:
            Tensor of shape (1, max_l) or (B, max_l).
        """
        ...

    @abstractmethod
    def dp_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        """
        Time-derivative d(p_mask)/dt at timestep t.

        Used in the ELBO loss weight dp/p.  The sign convention matches
        p_mask: positive values mean masking probability increases with t.

        Args:
            t:      Timestep — float, 0-dim tensor, or (B, 1) tensor.
            max_l:  Sequence length (number of positions).
            device: Target device for the returned tensor.

        Returns:
            Tensor of shape (1, max_l) or (B, max_l).
        """
        ...

    def plot(self, max_l: int, device: torch.device, save_path: str, n_samples: int = 5) -> None:
        timesteps = np.linspace(0, 1, n_samples)
        positions = np.arange(max_l)

        fig, axes = plt.subplots(n_samples, 1, figsize=(8, 2 * n_samples), sharex=True)
        fig.suptitle(f"{self.__class__.__name__} — masking probability vs position")
        for ax, t_val in zip(axes, timesteps):
            t = torch.tensor(t_val, dtype=torch.float32, device=device)
            p = self.p_mask(t, max_l, device).squeeze(0).cpu().numpy()
            ax.plot(positions, p)
            ax.set_ylabel("p_mask")
            ax.set_ylim(0, 1)
            ax.set_title(f"t={t_val:.2f}", fontsize=9)
        axes[-1].set_xlabel("Position")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
