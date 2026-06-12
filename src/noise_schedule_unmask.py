import torch
import torch.nn as nn
import numpy as np
from constants import EOS_token, SOS_token, PAD_token, MASK_token

# Producing sampled tokens using vectorization
class ScheduledUnmasker(nn.Module):
    def __init__(self, model, device, T=100, denoise="0", oracle=False, oracle_model=None,
                 schedule=None, gaussian_noise=False, sigma=1.0):
        """
        Args:
            schedule:      A NoiseSchedule instance.  When provided it is used for
                           all alpha_t / alpha_s computations and the legacy
                           gaussian_noise / sigma flags are ignored.
            gaussian_noise: Legacy flag — kept for backward compatibility.
            sigma:          Legacy Gaussian sigma — kept for backward compatibility.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.T = T
        self.denoise = denoise
        self.oracle = oracle
        # Optional oracle model for parallel validation when the main model is NOT the oracle.
        # Must expose a .validate(X) method returning (bool, error_str_or_None).
        self.oracle_model = oracle_model

        # Build schedule from legacy flags when no explicit schedule is provided.
        if schedule is not None:
            self._schedule = schedule
        elif gaussian_noise:
            from schedules.gaussian_schedule import GaussianSchedule
            self._schedule = GaussianSchedule(sigma)
        else:
            from schedules.categorical_schedule import CategoricalSchedule
            self._schedule = CategoricalSchedule()

        # Keep legacy attributes so existing code that reads them still works.
        self.gaussian_noise = gaussian_noise
        self.sigma = sigma

    # fraction (0 <= fr <= 1) specifies the next step 
    def forward(self, init_X, timestep, strategy = 'categorical', temperature=1.0, return_steps=False, eps=1e-5):
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

        steps, timesteps_log = [X.clone()], [timestep]
                
        if not self.oracle:
            self.model.eval()
        
        error_message = 'No errors occured.'
        error_probs, error_logits, error_changed_mask = None, None, None
        
        with torch.no_grad():            
            for i in range(num_steps):
                if timesteps[i] <= 0:
                    break
                # Linear schedule: α_t = 1 - t, where α_t is the propotion of original content retained at step t.
                # t = 0 (clean data) => α_t = 1, t = 1 (fully masked) => α_t = 0
                # s < t => α_s > α_t => more content retained at step s than t.
                
                # Compute retention probabilities via the noise schedule.
                # Both schedules return (1, L); the categorical schedule
                # broadcasts the scalar timestep to match position count.
                alpha_t = 1 - self._schedule.p_mask(timesteps[i], max_l=L, device=self.device)
                alpha_s = 1 - self._schedule.p_mask(timesteps[i] - dt, max_l=L, device=self.device)
                
                if not self.oracle:
                    # Get model predictions
                    logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]  # (L, 6)
                    
                    if self.oracle_model is not None:
                        try:
                            self.oracle_model.forward(X)
                        except ValueError as e:
                            changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist() if error_changed_mask is not None else []
                            error_message = f'''Oracle failed at step {i} with input {X}.
                            Message:{e}
                            Investigating probs and logits:
                            '''
                            for token in changed_tokens:
                                error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'
                            break
                else:
                    try:
                        logits = self.model(X)
                    except ValueError as e:
                        changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist() if error_changed_mask is not None else []
                        error_message = f'''Oracle failed at step {i} with input {X}.
                        Message:{e}
                        Investigating probs and logits:
                        '''
                        for token in changed_tokens:
                            error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'
                        break
                
                # Convert to probabilities (x_θ in the paper)
                scaled_logits = logits[:, :-1] / temperature if temperature > 0 else logits[:, :-1]
                if self.oracle:
                    content_probs = scaled_logits  # already probabilities
                else:
                    content_probs = torch.softmax(scaled_logits, dim=-1)
                
                probs = torch.zeros_like(logits)

                weight = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
                mask_prob = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0)
                # alpha tensors are (1, L); reshape for correct broadcast with (L, vocab-1)
                weight = weight.squeeze(0).unsqueeze(-1)   # (L, 1)
                mask_prob = mask_prob.squeeze(0)            # (L,)
                probs[:, :-1] = content_probs * weight
                probs[:, -1] = mask_prob
                # Positions where p_mask_t=0 produce 0/0=NaN; they are unmasked in X so
                # won't be used, but multinomial requires every row to be valid.
                probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                zero_rows = probs.sum(dim=-1) == 0
                probs[zero_rows, -1] = 1.0  # fallback: sample MASK (position stays unmasked)
                
                if temperature <= 0:
                    # greedy sampling
                    sampled_X = probs.argmax(dim=-1)
                else:
                    sampled_X = torch.multinomial(probs, 1).squeeze(-1)
                
                error_probs, error_logits, error_changed_mask = probs, logits, ((X == MASK_token) & (sampled_X != MASK_token))

                X[X == MASK_token] = sampled_X[X == MASK_token]
                steps.append(X.clone())
                timesteps_log.append(timesteps[i] - dt)

            # Ensure no more MASK tokens remain (can happen due to numerical issues with the noise schedule)
            if (X == MASK_token).any():
                if self.oracle:
                    try:
                        logits = self.model(X)
                        do_mopup = True
                    except ValueError:
                        do_mopup = False
                else:
                    do_mopup = True
                    logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]

                if do_mopup:
                    probs = torch.softmax(logits[:, :-1], dim=-1)
                    if temperature <= 0:
                        sampled_X = probs.argmax(dim=-1)
                    else:
                        sampled_X = torch.multinomial(probs, 1).squeeze(-1)
                    X[X == MASK_token] = sampled_X[X == MASK_token]
                    steps.append(X.clone())
                    timesteps_log.append(timesteps[-1] - dt)
            
            if return_steps == True:
                return X, steps, timesteps_log, error_message
            return X 