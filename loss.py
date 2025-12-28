import torch
import torch.nn as nn

class rblb(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, X, logits, y_true, timestep): 
        timestep = torch.clamp(timestep, min=0.01, max=1.0)  # No t < 0.01
        B, L = X.shape
        # X (B, L); float32
        # logits (B, L, 2); float32
        # y_true (B, L); long
        
        loss = self.loss_fn(logits.view(B*L, -1), y_true.view(B*L))
        loss = loss.view((B, L))
        
        mask = (X==2).float()
        loss *= mask

        # calculating weighted loss
        loss = loss.reshape((B, L))
        loss = 1.0/(timestep.unsqueeze(-1) + 1e-5) * loss
        loss = loss.sum() / B

        return loss
