import torch
import torch.nn as nn

class rblb(nn.Module):
    def __init__(self, device):
        super(rblb, self).__init__()

        self.device = device
        
    def forward(self, X, y_pred, y_true, timestep):        
        # X (B, L); float32
        # y_pred (B, 2, L); float32
        # y_true (B, L); long

        # hardcode: removing guesses for unmasked tokens -> set Loss over these tokens to 0
        global_mask = (X.to(torch.long) == 2).to(torch.float32)
        f = nn.functional.cross_entropy(y_pred, y_true, reduction="none") * global_mask

        # calculating weighted loss
        f = 1/(timestep + 1e-4) * f.sum(dim=-1)

        return f.mean()
