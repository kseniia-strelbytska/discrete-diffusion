import torch
import torch.nn as nn
import numpy as np
from constants import EOS_token, SOS_token, PAD_token, MASK_token

# Producing sampled tokens using vectorization
class ScheduledUnmasker(nn.Module):
    def __init__(self, model, device, T=100):
        super().__init__()
        self.model = model
        self.device = device
        self.T = T

    # fraction (0 <= fr <= 1) specifies the next step 
    def forward(self, init_X, timestep, eps=1e-5, return_steps=False):
        X = init_X.clone().long().to(self.device)
        L = X.shape[0]
        
        # scale down the number of denoising steps acc to noise level
        num_steps = int(self.T * timestep)
        timesteps = torch.linspace(timestep, eps, num_steps)
        dt = (timestep - eps) / num_steps
        
        steps = [X.clone()]
                
        self.model.eval()
        with torch.no_grad():            
            for i in range(num_steps):
                # Linear schedule: α_t = 1 - t
                alpha_t = 1 - timesteps[i]
                alpha_s = 1 - (timesteps[i] - dt)
                
                # Get model predictions
                logits = self.model(X.unsqueeze(0))[0]  # (L, 6)
                
                # Convert to probabilities (x_θ in the paper)
                probs = torch.softmax(logits, dim=-1)  # (L, 6)
                
                probs[:, :-1] *= (alpha_s - alpha_t) / (1 - alpha_t)
                probs[:, -1] = (1 - alpha_s) / (1 - alpha_t) # mask prob
                                                
                sampled_X = torch.distributions.categorical.Categorical(probs=probs).sample()
                
                X[X == MASK_token] = sampled_X[X == MASK_token]
                steps.append(X.clone())

            if return_steps == True:
                return X, steps
            return X 