import torch
import torch.nn as nn
from constants import EOS_token, SOS_token, PAD_token, MASK_token

from gaussian.gaussian_noise_schedule import get_gaussian_noise_schedule

class GaussianLoss(nn.Module):
    def __init__(self, device, vocab_size, T, sigma=1.0, sampling_eps=1e-5, eos_weight=10.0, inverse_t=False):
        super().__init__()
        
        self.neg_infinity = -1000000.0
        self.device=device
        self.T = T
        self.sigma = sigma
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t
        self.eos_weight = eos_weight
        
        # class_weight = torch.tensor([1.0] * vocab_size, device=device)
        # class_weight[EOS_token] = eos_weight

        # self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=class_weight)
        # self.loss_fn = self.loss_fn.to(device)
        
    def subs_parameterisation(self, logits, xt):
        '''
        Takes model output logits (B, L, vocab_size)
        xt = sample at time t 
        Returns ans, where ans.exp() = logits sub parameterised
        '''
       
        logits[:, :, MASK_token] = self.neg_infinity
        
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        
        unmasked_indices = (xt != MASK_token)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        
        return logits
    
    # ELBO loss, (eq 8 in MDLM paper)
    def forward(self, xt, logits, y_true, timestep):
        # eq10
        logits = logits.clone()
        logits = self.subs_parameterisation(logits, xt)
        
        p_mask, p_mask_dt = get_gaussian_noise_schedule(t_i=timestep, sigma=self.sigma, max_l=logits.shape[1], device=self.device)
        p_mask = p_mask.clamp(self.sampling_eps, 1 - self.sampling_eps)

        #Extracts the logits corresponding to the true class labels
        log_prob_x = torch.gather(logits, -1, y_true[:, :, None]).squeeze(-1) 
        loss = -p_mask_dt / p_mask * (-log_prob_x)
        
        # print(f'Sanity check: p_mask_dt={p_mask_dt}, p_mask={p_mask}, log_prob_x={log_prob_x}')
        # print(f'Sanity check: logits contains NaN: {torch.isnan(logits).any()}')

        # upweight EOS targets for both eq8 and eq9
        token_weight = torch.ones_like(loss).to(loss.device)
        token_weight = token_weight.masked_fill(y_true == EOS_token, self.eos_weight)
        loss = loss * token_weight
        
        loss = loss.sum() / torch.numel(xt) # average over all tokens
        
        return loss