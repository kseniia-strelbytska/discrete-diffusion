import torch
import numpy as np

from constants import MASK_token, PAD_token, SOS_token

class Dataset(torch.utils.data.Dataset):
    """
    If inverse_t is True, samples a random masking probability for each sample from 1/x.
    
    Dependent on the number of timesteps (T)
    """

    def __init__(self, y_data, device, T, sampling_eps=1e-5, inverse_t=False, suffix_mix_prob=0.0):
        self.device = device
        self.y_data = y_data.to(device)

        self.T = T
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t
        self.suffix_mix_prob = suffix_mix_prob

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

        # T=0 signals autoregressive mode: no noise schedule, return clean sequences.
        if self.T == 0:
            timestep = torch.zeros(y_batch.shape[0], 1, device=self.device)
            return y_batch, y_batch, timestep
        
        prob = None
        if self.inverse_t:
            prob = self.sample_inverse_t(y_batch.shape[0])
        else:
            prob = self.stratified_sampling(y_batch.shape[0])
            
        # Random masking: every content position independently masked with probability t
        rand_mask = (torch.rand_like(y_batch, dtype=torch.float, device=self.device) < prob) \
                    & (y_batch != PAD_token) \
                    & (y_batch != SOS_token)

        if self.suffix_mix_prob > 0.0:
            # Suffix masking: reveal the first (1-t) fraction of content, mask the rest.
            # This trains the model on prefix-completion (count and predict), directly
            # matching the evaluation distribution (e.g. "limited" / "complete" prompts).
            is_content = (y_batch != PAD_token) & (y_batch != SOS_token)   # (B, L)
            content_lengths = is_content.sum(dim=1).long()                  # (B,)
            prob_1d = prob.squeeze(-1)                                       # (B,)
            n_mask = (prob_1d * content_lengths.float()).round().long().clamp(min=0)
            split_lengths = content_lengths - n_mask   # revealed content tokens per row
            content_cumsum = is_content.long().cumsum(dim=1)               # (B, L)
            # A position is in the suffix iff it is a content position AND its content
            # index (1-based) exceeds the number of revealed tokens.
            suffix_mask = is_content & (content_cumsum > split_lengths.unsqueeze(1))

            # Per-item choice: suffix masking (True) or random masking (False)
            use_suffix = (torch.rand(y_batch.shape[0], device=self.device) < self.suffix_mix_prob)
            mask = torch.where(use_suffix.unsqueeze(1), suffix_mask, rand_mask)
        else:
            mask = rand_mask

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