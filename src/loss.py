import torch
import torch.nn as nn
from constants import EOS_token, SOS_token, PAD_token, MASK_token

class rblb(nn.Module):
    def __init__(self, eos_weight=10.0, inverse_t=False, device='cpu'):
        super().__init__()
        
        self.inverse_t = inverse_t
        
        class_weight = torch.tensor([1.0] * 5)
        class_weight[EOS_token] = eos_weight

        self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=class_weight)
        self.loss_fn = self.loss_fn.to(device)
        
    def forward(self, X, logits, y_true, timestep): 
        timestep = torch.clamp(timestep, min=0.01, max=1.0)  # No t < 0.01
        B, L = X.shape
        # X (B, L); float32
        # logits (B, L, 2); float32
        # y_true (B, L); long
        
        loss = self.loss_fn(logits.view(B*L, -1), y_true.view(B*L))
        loss = loss.view((B, L))
        
        mask = (X==MASK_token).float()
        loss *= mask

        # calculating weighted loss
        loss = loss.reshape((B, L))
        if self.inverse_t:
            loss = 1.0/(timestep.unsqueeze(-1) + 1e-5) * loss
        
        loss = loss.sum() / B

        return loss
