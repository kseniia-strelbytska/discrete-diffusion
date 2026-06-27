import torch
import numpy as np

from .constants import MASK_token


class Dataset(torch.utils.data.Dataset):
    def __init__(self, y_data, device, T, sampling_eps=1e-5, inverse_t=False):
        self.device = device
        self.y_data = y_data.to(device)
        self.T = T
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, index):
        return self.y_data[index]

    def stratified_sampling(self, batch_size):
        shift = torch.rand((batch_size, ), device=self.device) / batch_size
        samples = torch.arange(batch_size, device=self.device) / batch_size + shift
        samples %= 1
        samples = (1 - self.sampling_eps) * samples + self.sampling_eps
        samples = (samples * self.T).to(torch.int)
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

        if self.T == 0:
            timestep = torch.zeros(y_batch.shape[0], 1, device=self.device)
            return y_batch, y_batch, timestep

        prob = None
        if self.inverse_t:
            prob = self.sample_inverse_t(y_batch.shape[0])
        else:
            prob = self.stratified_sampling(y_batch.shape[0])

        mask = torch.rand_like(y_batch, dtype=torch.float, device=self.device) < prob
        x_batch = torch.where(mask, torch.full_like(y_batch, MASK_token), y_batch)

        return x_batch, y_batch, prob


def get_fixed_dataset(dataset, device, batch_size=32):
    fixed_dataset = []
    y_samples = []

    for y_sample in dataset:
        y_samples.append(y_sample)

        if len(y_samples) == batch_size:
            x, y, prob = dataset.masking_collate_fn(y_samples)
            fixed_dataset.append((x, y, prob))
            y_samples = []

    if y_samples:
        x, y, prob = dataset.masking_collate_fn(y_samples)
        fixed_dataset.append((x, y, prob))

    return fixed_dataset
