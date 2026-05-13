import torch
from torch import Tensor

from schedules.noise_schedule import NoiseSchedule


class CategoricalSchedule(NoiseSchedule):
    """
    Uniform (position-independent) linear masking schedule.

    Each position is masked independently with probability t, so:

        p_mask(t, i)  = t   for all positions i
        dp_mask(t, i) = 1   for all positions i

    This is the schedule used by MDLM / SUBS-parameterisation with a
    categorical prior.  The resulting ELBO loss weight dp/p = 1/t matches
    the standard eq8 loss in loss.py.
    """

    def p_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        ones = torch.ones(1, max_l, device=device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        return t.to(device=device, dtype=torch.float32) * ones  # (B, max_l) or (1, max_l)

    def dp_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] > 1:
            return torch.ones(t.shape[0], max_l, device=device)
        return torch.ones(1, max_l, device=device)
