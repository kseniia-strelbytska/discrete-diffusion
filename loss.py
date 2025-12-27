import torch
import torch.nn as nn

class rblb(nn.Module):
    def __init__(self):
        super(rblb, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, X, logits, y_true, timestep):   
        B, L = X.shape
        # X (B, L); float32
        # logits (B, L, 2); float32
        # y_true (B, L); long
        
        loss = self.loss_fn(logits.view(B*L, -1), y_true.view(B*L))
        mask = (X==2).view(B*L).float()
        loss *= mask
        loss = torch.sum(loss) / torch.sum(mask)
        
        return loss

        # calculating weighted loss
        loss = loss.reshape((B, L))
        loss = 1/(timestep.unsqueeze(-1) + 1e-4) * loss
        loss = torch.sum(loss) / torch.sum(mask)

        return loss
