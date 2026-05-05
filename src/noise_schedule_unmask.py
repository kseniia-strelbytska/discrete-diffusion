import torch
import torch.nn as nn
import numpy as np
from constants import EOS_token, SOS_token, PAD_token, MASK_token
from gaussian.gaussian_noise_schedule import get_gaussian_noise_schedule

# Producing sampled tokens using vectorization
class ScheduledUnmasker(nn.Module):
    def __init__(self, model, device, T=100, denoise="0", oracle=False, oracle_model=None, gaussian_noise=False, sigma=1.0):
        super().__init__()
        self.model = model
        self.device = device
        self.T = T
        self.denoise = denoise
        self.oracle = oracle
        # Optional oracle model for parallel validation when the main model is NOT the oracle.
        # Must expose a .validate(X) method returning (bool, error_str_or_None).
        self.oracle_model = oracle_model
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
                
                if self.gaussian_noise:
                    p_mask_t, _ = get_gaussian_noise_schedule(t_i=timesteps[i], sigma=self.sigma, max_l=L, device=self.device)
                    p_mask_s, _ = get_gaussian_noise_schedule(t_i=timesteps[i] - dt, sigma=self.sigma, max_l=L, device=self.device)
                    
                    alpha_t = 1 - p_mask_t
                    alpha_s = 1 - p_mask_s
                else:
                    alpha_t = 1 - timesteps[i]
                    alpha_s = 1 - (timesteps[i] - dt)
                
                if not self.oracle:
                    # Get model predictions
                    logits = self.model(X.unsqueeze(0), timesteps[i].unsqueeze(0))[0]  # (L, 6)
                    
                    if self.oracle_model is not None:
                        oracle_result = self.oracle_model.forward(X)
                        if oracle_result[0] is None:
                            
                            if error_changed_mask is not None:
                                changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist()
                            else:
                                changed_tokens = []
                            
                            error_message = f'''Oracle failed at step {i} with input {X}.
                            Message:{oracle_result[1]}
                            Investigating probs and logits:
                            '''
                            for token in changed_tokens:
                                error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'
                            break
                else:
                    logits = self.model(X)
                    if logits[0] == None:
                        # If oracle returns None, it means the input cannot be completed correctly.
                        
                        changed_tokens = error_changed_mask.nonzero(as_tuple=True)[0].tolist() if error_changed_mask is not None else []
                        
                        error_message = f'''Oracle failed at step {i} with input {X}.
                        Message:{logits[1]}
                        Investigating probs and logits:
                        '''
                        for token in changed_tokens:
                            error_message += f'\nToken index {token}\nprob={error_probs[token].cpu().numpy()}\nlogit={error_logits[token].cpu().numpy()}\nChoice: {X[token].item()}\n'

                        break
                    logits = logits[1]

                # Convert to probabilities (x_θ in the paper)
                if temperature <= 0: # greedy
                    content_probs = torch.softmax(logits[:, :-1], dim=-1)
                else:
                    content_probs = torch.softmax(logits[:, :-1] / temperature, dim=-1)
                
                probs = torch.zeros_like(logits)

                weight = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
                mask_prob = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0)
                if self.gaussian_noise:
                    # alpha tensors are (1, L); reshape for correct broadcast with (L, vocab-1)
                    weight = weight.squeeze(0).unsqueeze(-1)   # (L, 1)
                    mask_prob = mask_prob.squeeze()             # (L,)
                probs[:, :-1] = content_probs * weight
                probs[:, -1] = mask_prob
                # Positions where p_mask_t=0 produce 0/0=NaN; they are unmasked in X so
                # won't be used, but multinomial requires every row to be valid.
                probs = probs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                zero_rows = probs.sum(dim=-1) == 0
                probs[zero_rows, -1] = 1.0  # fallback: sample MASK (position stays unmasked)

                # probs[:, :-1] *= ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(min=0.0)
                # probs[:, -1] = ((1 - alpha_s) / (1 - alpha_t)).clamp(min=0.0, max=1.0) # mask prob
                
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
                
                error_probs, error_logits, error_changed_mask = probs, logits, ((X == MASK_token) & (sampled_X != MASK_token))

                X[X == MASK_token] = sampled_X[X == MASK_token]
                steps.append(X.clone())
                timesteps_log.append(timesteps[i] - dt)

            if return_steps == True:
                return X, steps, timesteps_log, error_message
            return X 