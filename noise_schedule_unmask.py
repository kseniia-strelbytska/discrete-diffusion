import torch
import torch.nn as nn
import numpy as np

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
            
            while timestep > 0.00 and (X == 2).sum() > 0:
                t = timestep
                s = max(0.0, t - (1/T if timestep > 0.01 else 0.001))
                timestep = s
                
                # Linear schedule: α_t = 1 - t
                alpha_t = 1 - t
                alpha_s = 1 - s
                
                # Get model predictions
                logits = self.model(X.unsqueeze(0))[0]  # (L, 3)
                
                # Convert to probabilities (x_θ in the paper)
                probs = torch.softmax(logits, dim=-1)  # (L, 3)
                
                # For each position independently
                for pos in range(L):
                    if X[pos] == 2:  # If masked
                        # Create transition distribution
                        transition_probs = torch.zeros(3, device=X.device)
                        
                        # p(z_s = k) = (α_s - α_t)/(1-α_t) * x_θ^k  for k ≠ MASK
                        unmask_coeff = (alpha_s - alpha_t) / (1 - alpha_t)
                        transition_probs[0] = unmask_coeff * probs[pos, 0]
                        transition_probs[1] = unmask_coeff * probs[pos, 1]
                        
                        # p(z_s = MASK) = (1 - α_s)/(1-α_t)
                        stay_masked_prob = (1 - alpha_s) / (1 - alpha_t)
                        transition_probs[2] = stay_masked_prob
                        
                        # Sample from this distribution
                        X[pos] = torch.multinomial(transition_probs, 1).item()
                    
                    # If not masked, it stays the same (carry-over)
            return X 

class SequencedScheduledUnmasker(nn.Module):
    def __init__(self, model, fraction):
        super().__init__()
        self.model = model 
        self.unmasker_model = ScheduledUnmasker(model, fraction)

        self.fraction = fraction

    def forward(self, X, timestep):
        X_unmasked = X.clone()
        for _ in range(40):
            X_unmasked = self.unmasker_model(X_unmasked, timestep)

        return X_unmasked

# def get_scheduled_unmasker(model, fraction):
#     return nn.Sequential(
#         *[ScheduledUnmasker(model, fraction) for _ in range(40)],
#         ScheduledUnmasker(model, 1.0)
#     )

