import torch
import torch.nn as nn

class rblb(nn.Module):
    def __init__(self):
        super(rblb, self).__init__()
        
    def forward(self, y_pred, y_true, alpha):
        return (nn.functional.cross_entropy(y_pred, y_true, reduction="none").mean(dim=1) * alpha).sum()