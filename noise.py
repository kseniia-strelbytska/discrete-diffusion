import torch

def add_noise(X, alpha=0.1):
    noise = torch.rand_like(X) < alpha

    X[noise] = 2 

    return X

def generate_noise_seqs(X):
    seqs = X.clone()[:, None, :]

    while torch.isclose(X, torch.tensor([2], dtype=torch.float32)).sum() != X.size(dim=0) * X.size(dim=1):
        X = add_noise(X)

        seqs = torch.cat((seqs, X[:, None, :]), 1)
    return seqs