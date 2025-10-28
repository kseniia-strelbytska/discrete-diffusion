import torch
import torch.nn as nn

class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)