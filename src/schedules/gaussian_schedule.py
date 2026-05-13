import torch
from torch import Tensor

from schedules.noise_schedule import NoiseSchedule
from gaussian.gaussian_noise_schedule import get_gaussian_noise_schedule


class GaussianSchedule(NoiseSchedule):
    """
    Gaussian (CDF-based) noise schedule with a soft spatial masking boundary.

    At each timestep t the masking probability at position i is the CDF of a
    Normal distribution whose mean sweeps from right (t=0, nothing masked) to
    left (t=1, everything masked).  sigma controls how sharp the boundary is.

    This wraps the existing get_gaussian_noise_schedule function so that it
    participates in the NoiseSchedule contract.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def p_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        p, _ = get_gaussian_noise_schedule(t, self.sigma, max_l, device)
        return p

    def dp_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        _, dp = get_gaussian_noise_schedule(t, self.sigma, max_l, device)
        return dp
