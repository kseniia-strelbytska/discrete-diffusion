import torch
import itertools
import noise
import random
import numpy as np

def generate_seq(length):
    idxs = torch.arange(length).tolist()

    combs = list(itertools.combinations(idxs, length // 2))

    data = torch.zeros(len(combs), length).long()
    data[torch.arange(len(combs))[:, None], combs] = 1

    return data

def gen_data(length, prob):
    all_good_seq = generate_seq(length)
    seqs = all_good_seq[torch.rand(all_good_seq.shape[0]) < prob]
    seqs = noise.generate_noise_seqs(seqs)
    out = []
    for x in seqs:
        x = torch.unique_consecutive(x, dim=0)
        for y in x:
            out.append([list(y), list(x[0])])
    return torch.Tensor(out).long()

def sample_uniform_t(batch_size):
    # sample time t between 0 and 1 (t = fraction of tokens that are masked)
    t = np.random.uniform(0, 1, (batch_size))

    return torch.tensor(t)

def sample_inverse_t(batch_size):
    # assume area between 0.01 and 1
    total_area = np.log(1) - np.log(0.01)
    sampled_area = np.random.uniform(0.01, total_area, (batch_size))

    sampled_t = []

    for area in sampled_area:
        l, r = 0.001, 1 

        while r - l > 0.001:
            mid = (l + r) / 2.0 

            point_area = np.log(mid) - np.log(0.01)

            if point_area < area:
                l = mid 
            else:
                r = mid

        sampled_t.append((l + r) / 2.0)

    return torch.tensor(sampled_t)

def sample_masked(length, batch_size, t):
    seqs = generate_seq(length)
    sampled_seqs = seqs[torch.randint(0, seqs.size(0), (batch_size,))]

    sampled_seqs = torch.where(torch.rand((batch_size, length)) < t[:, None], torch.full((batch_size, length), torch.tensor(2)), sampled_seqs)

    return sampled_seqs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, sample_prob):
        self.data = gen_data(length, sample_prob)
        self.length = length
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index][0].float(), torch.nn.functional.one_hot(self.data[index][1], 
               num_classes=2).float().permute(1, 0), torch.sum(self.data[index][0] != 2).float()/self.length

# ds = Dataset(20, 0.01)
# train_dataloader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)