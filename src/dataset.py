import torch
import numpy as np

from constants import MASK_token, EOS_token

class Dataset(torch.utils.data.Dataset):
    """
    masking_type controls how tokens are masked during training:
      "random"  – each token independently masked with probability t (MDLM default)
      "suffix"  – reveal a random prefix, mask the remaining suffix (better alignment
                  with LTR denoising at eval time; recommended for a^n b^n grammar)

    If inverse_t is True, samples a random masking probability for each sample from 1/x.
    Dependent on the number of timesteps (T)
    """

    def __init__(self, y_data, device, T, sampling_eps=1e-5, inverse_t=False, masking_type="random"):
        self.device = device
        self.y_data = y_data.to(device)
        
        self.T = T
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t
        self.masking_type = masking_type

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
        u = np.random.uniform(0, 1, size=(batch_size, ))
        log_clip = np.log(CLIP_VALUE)
        sampled_val = np.exp(u * (np.log(1.0) - log_clip) + log_clip)

        return torch.tensor(sampled_val, device=self.device)

    def masking_collate_fn(self, y_batch):
        y_batch = torch.stack(y_batch).to(self.device)
        B, L = y_batch.shape

        # T=0 signals autoregressive mode: no noise schedule, return clean sequences.
        if self.T == 0:
            timestep = torch.zeros(B, 1, device=self.device)
            return y_batch, y_batch, timestep

        if self.masking_type == "suffix":
            # Suffix masking: reveal a random contiguous prefix, mask the rest.
            # This aligns training with LTR denoising at eval time (a^n b^n key insight:
            # the model always sees the full run of 0s before predicting 1s).
            prob = self.stratified_sampling(B)  # (B, 1) fraction of tokens to mask
            # Number of tokens to KEEP revealed in [1, L-1] (SOS always visible)
            split = ((1.0 - prob) * L).long().clamp(1, L - 1)  # (B, 1)
            positions = torch.arange(L, device=self.device).unsqueeze(0)  # (1, L)
            mask = positions >= split  # (B, L) – True where to mask
            x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)
            return x_batch, y_batch, prob

        if self.masking_type == "grammar_suffix":
            # Grammar-aware suffix masking for a^n b^n:
            # ALWAYS reveal all zeros (full counting context), then mask k..n ones + EOS.
            # Split point is in [ones_start, eos_pos] → model always sees all n zeros,
            # then some k ones (k uniform in [0, n]), and must predict (n-k) ones + EOS.
            # This EXACTLY matches the evaluation distribution (complete dataset prompts).

            # Find position of first '1' token (grammar token value=1)
            ones_start = (y_batch == 1).long().argmax(dim=1)      # (B,)
            # Find EOS position
            eos_pos = (y_batch == EOS_token).long().argmax(dim=1)  # (B,)

            # n = number of ones in the sequence = eos_pos - ones_start
            n_ones = (eos_pos - ones_start).clamp(min=0)  # (B,)

            # Sample number of ones to REVEAL: k uniform in {0, 1, ..., n}
            # Use continuous uniform and floor for discrete sampling
            u = torch.rand(B, device=self.device)  # (B,)
            k = (u * (n_ones + 1).float()).long()
            k = torch.minimum(k, n_ones)  # (B,) – clamp to [0, n]

            # Split: reveal SOS + all zeros + k ones
            split = (ones_start + k).unsqueeze(1).clamp(1, L - 1)  # (B, 1)

            # Timestep = fraction of tokens masked = (L - split) / L
            prob = ((L - split.float()) / L).clamp(self.sampling_eps, 1 - self.sampling_eps)

            positions = torch.arange(L, device=self.device).unsqueeze(0)  # (1, L)
            mask = positions >= split  # (B, L)
            x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)
            return x_batch, y_batch, prob

        # Default: independent Bernoulli masking (MDLM / random order)
        prob = None
        if self.inverse_t:
            prob = self.sample_inverse_t(B)
        else:
            prob = self.stratified_sampling(B)

        mask = torch.rand_like(y_batch, dtype=torch.float, device=self.device) < prob
        x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)

        return x_batch, y_batch, prob

def get_fixed_dataset(dataset, device, batch_size=32):
    fixed_dataset = []
    x_samples, y_samples, timesteps = [], [], []

    for y_sample in dataset:
        # Apply masking to individual sample
        
        y_samples.append(y_sample)
        
        # When we have enough samples for a batch, stack them and add to dataset
        if len(y_samples) == batch_size:
            x, y, prob = dataset.masking_collate_fn(y_samples)
            fixed_dataset.append(
                (x, y, prob)
            )
            y_samples = []

    # Handle remaining samples if any
    if y_samples:
        x, y, prob = dataset.masking_collate_fn(y_samples)
        fixed_dataset.append(
            (x, y, prob)
        )

    return fixed_dataset