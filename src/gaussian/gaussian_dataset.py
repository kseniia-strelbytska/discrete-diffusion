import torch
import numpy as np
import math

from constants import MASK_token
from gaussian.gaussian_noise_schedule import get_gaussian_noise_schedule

class GaussianDataset(torch.utils.data.Dataset):
    """
    If inverse_t is True, samples a random masking probability for each sample from 1/x.
    
    Dependent on the number of timesteps (T)
    """

    def __init__(self, y_data, device, T, sigma, max_l, sampling_eps=1e-5, inverse_t=False):
        self.device = device
        self.y_data = y_data.to(device)
        
        self.T = T
        self.sampling_eps = sampling_eps
        self.sigma = sigma
        self.max_l = max_l
        self.inverse_t = inverse_t

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, index):
        return self.y_data[index]

    # @staticmethod
    # def apply_masking(y_batch, prob_batch, device):
    #     mask = torch.rand_like(y_batch, dtype=torch.float, device=device) < prob_batch
    #     x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)

    #     return x_batch

    def stratified_sampling(self, batch_size):
        shift = torch.rand((batch_size, ), device=self.device) / batch_size
        samples = torch.arange(batch_size, device=self.device) / batch_size + shift 
        samples %= 1 
        samples = (1 - self.sampling_eps) * samples + self.sampling_eps
        
        samples = (samples * self.T).to(torch.int) # round down
        samples = (samples.float() / self.T) + 1 / self.T
        
        return samples.unsqueeze(-1)

    def sample_inverse_t(self, batch_size):
        CLIP_VALUE = 1e-2
        log_clip = math.log(CLIP_VALUE)
        u = torch.rand((batch_size, ), device=self.device)
        sampled_val = torch.exp(u * (math.log(1.0) - log_clip) + log_clip)

        return sampled_val.unsqueeze(-1) # Shape: (batch_size, 1)
    
    def masking_collate_fn(self, y_batch):
        y_batch = torch.stack(y_batch).to(self.device)

        # T=0 signals autoregressive mode: no noise schedule, return clean sequences.
        if self.T == 0:
            timestep = torch.zeros(y_batch.shape[0], 1, device=self.device)
            return y_batch, y_batch, timestep
        
        timesteps = None
        if self.inverse_t:
            timesteps = self.sample_inverse_t(y_batch.shape[0])
        else:
            timesteps = self.stratified_sampling(y_batch.shape[0])
        
        p_mask, _ = get_gaussian_noise_schedule(t_i=timesteps, sigma=self.sigma, max_l=self.max_l, device=self.device)
                
        mask = torch.rand_like(y_batch, dtype=torch.float, device=self.device) < p_mask
        x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)
                    
        return x_batch, y_batch, timesteps