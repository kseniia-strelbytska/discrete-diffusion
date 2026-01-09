import torch
import torch.nn as nn
import numpy as np
from constants import EOS_token, SOS_token, PAD_token, MASK_token

# Producing sampled tokens using vectorization
class ScheduledUnmasker(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # fraction (0 <= fr <= 1) specifies the next step 
    def forward(self, init_X, timestep):
        X = init_X.clone().long()
        L = X.shape[0]
        
        self.model.eval()
        with torch.no_grad():
            T = 100
            
            while timestep > 0.00 and (X == MASK_token).sum() > 0:
                t = timestep
                s = max(0.0, t - (1/T if timestep > 0.01 else 0.001))
                timestep = s
                
                # Linear schedule: α_t = 1 - t
                alpha_t = 1 - t
                alpha_s = 1 - s
                
                # Get model predictions
                logits = self.model(X.unsqueeze(0))[0]  # (L, 5)
                
                # Convert to probabilities (x_θ in the paper)
                probs = torch.softmax(logits, dim=-1)  # (L, 5)
                probs = torch.cat([probs, torch.full((L,1), torch.tensor(1))], dim=-1)
                
                probs[:, :5] *= (alpha_s - alpha_t) / (1 - alpha_t)
                probs[:, 5] = (1 - alpha_s) / (1 - alpha_t)
                                
                # sampled_X = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
                sampled_X = torch.distributions.categorical.Categorical(probs=probs).sample()
                
                X[X == MASK_token] = sampled_X[X == MASK_token]
                
            return X 