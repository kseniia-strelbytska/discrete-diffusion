import torch
import torch.nn as nn

class Unmasker(nn.Module):
    def __init__(self, model, alpha):
        super(Unmasker, self).__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, X):
        y_pred = self.model(X)

        mask = torch.rand_like(X) < self.alpha

        X_unmasked = X.clone()
        X_unmasked[torch.isclose(X, 2) & mask] = y_pred[torch.isclose(X, 2) & mask].argmax(dim=2)
        return X_unmasked
    
def get_unmasker(model):
    return nn.Sequential(
        *[Unmasker(model, 0.1) for _ in range(40)],
        Unmasker(model, 1.0)
    )