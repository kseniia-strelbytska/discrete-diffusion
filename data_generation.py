import torch
import itertools
import noise
import random

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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, sample_prob):
        self.data = gen_data(length, sample_prob)
        self.length = length
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index][0], torch.nn.functional.one_hot(self.data[index][1], 
               num_classes=2).float().permute(1, 0), torch.sum(self.data[index][0] != 2).float()/self.length

# ds = Dataset(20, 0.01)
# train_dataloader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)