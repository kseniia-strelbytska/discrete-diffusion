import torch
import itertools
import noise
import random

def generate_seq(length):
    idxs = torch.arange(length).tolist()

    combs = list(itertools.combinations(idxs, length // 2))

    data = torch.zeros(len(combs), length)
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
            out.append((y, x[0]))
    return out
