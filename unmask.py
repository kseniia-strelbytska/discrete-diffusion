import torch
import torch.nn as nn

class Unmasker(nn.Module):
    def __init__(self, model):
        super(Unmasker, self).__init__()
        self.model = model

    def forward(self, X, alpha=0.1):
        y_pred = self.model(X)

        mask = torch.rand_like(X) < alpha

        X_unmasked = X.clone()
        X_unmasked[torch.isclose(X, 2) & mask] = y_pred[torch.isclose(X, 2) & mask].argmax(dim=2)
        return X_unmasked
