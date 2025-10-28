import torch
import torch.nn as nn

class nll_loss(nn.Module):
    def __init__(self):
        super(nll_loss, self).__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)