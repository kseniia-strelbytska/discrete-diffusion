import torch
import torch.nn as nn

class ScheduledUnmasker(nn.Module):
    def __init__(self, model, fraction):
        super().__init__()
        self.model = model
        self.fraction = fraction
        self.device = model.device

    # fraction (0 <= fr <= 1) specifies the next step 
    def forward(self, X, timestep):
        X = X.to(self.device)

        self.model.eval()

        y_pred = self.model(X, timestep) # (B, 2, L)
        y_pred = torch.distributions.Categorical(logits=y_pred.permute(0, 2, 1)).sample()

        self.model.train()

        # count proportion of masked tokens
        # higher -> larger timestep
        timestep_t = (X == 2).sum(1) / X.size(1)
        alpha_t = 1 - timestep_t # prob of (un)masking a token

        # move one fraction step in the clean signal direction
        timestep_s = torch.minimum(torch.tensor(1), timestep_t - self.fraction)
        alpha_s = 1 - timestep_s 

        prob = (alpha_s - alpha_t) / (1 - alpha_t)
        mask = torch.rand_like(X, dtype=torch.float32) < prob

        X_unmasked = X.clone()        
        X_unmasked[(X == 2) & mask] = y_pred[(X == 2) & mask]

        return X_unmasked
    
def get_scheduled_unmasker(model, fraction):
    return nn.Sequential(
        *[ScheduledUnmasker(model, fraction) for _ in range(40)],
        ScheduledUnmasker(model, 1.0)
    )
