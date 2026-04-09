import torch
import torch.nn as nn
import numpy as np
from constants import EOS_token, SOS_token, PAD_token, MASK_token

# Producing sampled tokens using vectorization
class ScheduledUnmasker(nn.Module):
    def __init__(self, model, device, T=100, denoise="0"):
        super().__init__()
        self.model = model
        self.device = device
        self.T = T
        self.denoise = denoise

    # fraction (0 <= fr <= 1) specifies the next step
    def forward(self, init_X, timestep, strategy = 'categorical', temperature=1.0, return_steps=False, eps=1e-5):
        # Left-to-right denoising: special path, ignores timestep schedule
        if self.denoise == "ltr":
            result = self._ltr_forward(init_X, temperature, eps=eps)
            if return_steps:
                return result, [result]
            return result

        X = init_X.clone().long().to(self.device)
        timestep = timestep.clone().to(self.device)
        L = X.shape[0]

        # scale down the number of denoising steps acc to noise level
        if self.denoise == "eps":
            num_steps = int(self.T * timestep)
            timesteps = torch.linspace(timestep, eps, num_steps + 1, device=self.device)
            dt = (timestep - eps) / num_steps
        elif self.denoise == "0":
            #round timestep (up) to the nearest multiple of 1/T
            num_steps = int(torch.ceil(timestep * self.T).item())
            timestep = num_steps / self.T
            timesteps = torch.linspace(timestep, 0, num_steps + 1, device=self.device)
            dt = 1 / self.T
        else:
            raise ValueError(f"{self.denoise} is not defined")

        steps = [X.clone()]
                
        self.model.eval()
        with torch.no_grad():            
            for i in range(num_steps):
                # Linear schedule: α_t = 1 - t
                alpha_t = 1 - timesteps[i]
                alpha_s = 1 - (timesteps[i] - dt)
                
                # Get model predictions
                logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]  # (L, 6)
                
                # Convert to probabilities (x_θ in the paper)
                if temperature <= 0: # greedy
                    probs = torch.softmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                
                probs[:, :-1] *= (alpha_s - alpha_t) / (1 - alpha_t)
                probs[:, -1] = (1 - alpha_s) / (1 - alpha_t) # mask prob

                # if strategy == 'categorical':
                #     # sample from the categorical distribution
                #     sampled_X = torch.multinomial(probs, 1).squeeze(-1)
                # elif strategy == 'greedy':
                #     #greedy sampling
                #     sampled_X = probs.argmax(dim=-1)
                # else:
                #     raise ValueError(f"Unknown sampling strategy: {strategy}")
                
                if temperature <= 0:
                    # greedy sampling
                    sampled_X = probs.argmax(dim=-1)
                else:
                    sampled_X = torch.multinomial(probs, 1).squeeze(-1)
                                                
                #sampled_X = torch.distributions.categorical.Categorical(probs=probs).sample()
                
                #print(X[X != MASK_token])
                #print(sampled_X[X != MASK_token])

                X[X == MASK_token] = sampled_X[X == MASK_token]
                steps.append(X.clone())

            if return_steps == True:
                return X, steps
            return X

    def _ltr_forward(self, init_X, temperature, eps=1e-5):
        """
        Left-to-right denoising: unmask tokens from left to right.
        At each step the model sees all already-revealed tokens plus remaining MASKs,
        conditioned on the current noise level (= fraction of still-masked tokens).
        This mimics AR generation over the masked suffix while respecting the diffusion
        time conditioning, helping models with explicit timestep embeddings (e.g. DiT).
        """
        X = init_X.clone().long().to(self.device)

        # Collect masked positions in left-to-right order
        mask_positions = (X == MASK_token).nonzero(as_tuple=True)[0]
        n_total = X.numel()
        n_mask_start = len(mask_positions)

        if n_mask_start == 0:
            return X

        self.model.eval()
        with torch.no_grad():
            for step, pos in enumerate(mask_positions):
                # Timestep = fraction of tokens still masked — decreases as we reveal
                n_remaining = n_mask_start - step
                t = max(float(n_remaining) / n_total, eps)
                t_tensor = torch.tensor([t], dtype=torch.float, device=self.device)

                logits = self.model(X.unsqueeze(0), t_tensor)[0]  # (L, vocab_size)

                # Get logits for this position; suppress MASK token
                pos_logits = logits[pos].clone()
                pos_logits[MASK_token] = -1e9

                if temperature <= 0:
                    X[pos] = pos_logits.argmax()
                else:
                    probs = torch.softmax(pos_logits / temperature, dim=-1)
                    X[pos] = torch.multinomial(probs, 1).squeeze()

        return X