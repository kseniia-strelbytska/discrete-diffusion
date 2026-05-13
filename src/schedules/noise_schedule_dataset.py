import math

import torch
from torch import Tensor

from constants import MASK_token
from schedules.noise_schedule import NoiseSchedule


class NoiseScheduleDataset(torch.utils.data.Dataset):
    """
    Dataset for discrete diffusion training with an arbitrary noise schedule.

    At each training step a timestep t is sampled (stratified or inverse-t),
    then the schedule's p_mask is used to draw a position-wise Bernoulli mask
    over the clean sequence.

    The collate function returns (x_noisy, y_clean, timesteps):
      x_noisy  : (B, L) — clean tokens replaced by MASK_token with prob p_mask(t)
      y_clean  : (B, L) — original sequences (training targets)
      timesteps: (B, 1) — normalised timestep for each sample
    """

    def __init__(
        self,
        y_data: Tensor,
        device: torch.device,
        T: int,
        schedule: NoiseSchedule,
        max_l: int,
        sampling_eps: float = 1e-5,
        inverse_t: bool = False,
    ):
        self.device = device
        self.y_data = y_data.to(device)
        self.T = T
        self.schedule = schedule
        self.max_l = max_l
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t

    def __len__(self) -> int:
        return self.y_data.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.y_data[index]

    def stratified_sampling(self, batch_size: int) -> Tensor:
        """Uniform stratified timestep sampling in (sampling_eps, 1]."""
        shift = torch.rand((batch_size,), device=self.device) / batch_size
        samples = torch.arange(batch_size, device=self.device) / batch_size + shift
        samples %= 1
        samples = (1 - self.sampling_eps) * samples + self.sampling_eps
        samples = (samples * self.T).to(torch.int)
        samples = (samples.float() / self.T) + 1 / self.T
        return samples.unsqueeze(-1)  # (B, 1)

    def sample_inverse_t(self, batch_size: int) -> Tensor:
        """Sample timesteps from a 1/t distribution to up-weight early denoising."""
        CLIP_VALUE = 1e-2
        log_clip = math.log(CLIP_VALUE)
        u = torch.rand((batch_size,), device=self.device)
        sampled_val = torch.exp(u * (math.log(1.0) - log_clip) + log_clip)
        return sampled_val.unsqueeze(-1)  # (B, 1)

    def masking_collate_fn(self, y_batch) -> tuple:
        """
        Collate a list of clean sequences into a noisy training batch.

        T=0 signals autoregressive mode: returns clean sequences unchanged.
        """
        y_batch = torch.stack(y_batch).to(self.device)

        if self.T == 0:
            timestep = torch.zeros(y_batch.shape[0], 1, device=self.device)
            return y_batch, y_batch, timestep

        timesteps = (
            self.sample_inverse_t(y_batch.shape[0])
            if self.inverse_t
            else self.stratified_sampling(y_batch.shape[0])
        )  # (B, 1)

        p = self.schedule.p_mask(timesteps, max_l=self.max_l, device=self.device)
        mask = torch.rand_like(y_batch, dtype=torch.float, device=self.device) < p
        x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)

        return x_batch, y_batch, timesteps
