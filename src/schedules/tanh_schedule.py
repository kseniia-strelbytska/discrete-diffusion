import torch
from torch import Tensor

from schedules.noise_schedule import NoiseSchedule

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

    def get_dist(self, t_i : Tensor, sigma : float, max_l : int, device: torch.device) -> Tensor:
        """
        Generates a Gaussian distribution for a given timestep.

        Args:
            t_i (float): Normalized timestep in [0, 1]. (0 = clean data, 1 = full noise)
            sigma (float): The standard deviation (width of the soft boundary).
            max_l (int): The maximum length of the sequence. 
            device (torch.device): The device to place the tensors on.
        
        Returns:
            x (Tensor): A tensor of shape (1, max_l) containing position indices.
            width (float): The width parameter used for scaling the PDF to get dp/dt.
            dist (torch.distributions.Normal): The Normal distribution object for the given timestep.
        """
        x = torch.arange(max_l, dtype=torch.float32, device=device).unsqueeze(0)
        
        margin = 2.0 * sigma
        width = max_l + 2.0 * margin
        
        # At t=0 (clean), the mean is far right, CDF is ~0.
        # At t=1 (noisy), the mean is far left, CDF is ~1.
        mean = (1.0 - t_i) * width - margin
        dist = torch.distributions.Normal(mean, sigma)
        
        return x, width, dist

    def p_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        x, width, dist = self.get_dist(t, self.sigma, max_l, device)
        cdf = dist.cdf(x)
        p_mask = cdf 
        
        return p_mask
    def dp_mask(self, t: Tensor, max_l: int, device: torch.device) -> Tensor:
        x, width, dist = self.get_dist(t, self.sigma, max_l, device)
        pdf = dist.log_prob(x).exp()
        p_mask_dt = pdf * width

        return p_mask_dt
