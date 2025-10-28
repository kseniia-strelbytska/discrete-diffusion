import torch
import itertools

def generate_seq(length):
    idxs = torch.arange(length).tolist()

    combs = list(itertools.combinations(idxs, length // 2))

    data = torch.zeros(len(combs), 1, length)
    data[torch.arange(len(combs))[:, None], [0], combs] = 1

    return data
