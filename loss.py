import torch
import torch.nn as nn

class rblb(nn.Module):
    def __init__(self):
        super(rblb, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)