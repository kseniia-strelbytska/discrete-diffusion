import torch
import torch.nn as nn
from constants import EOS_token, SOS_token, PAD_token, MASK_token

class rblb(nn.Module):
    def __init__(self, device, vocab_size, T, sampling_eps=1e-5, eos_weight=10.0, inverse_t=False, loss_type="eq8"):
        super().__init__()
        
        self.neg_infinity = -1000000.0
        self.T = T
        self.sampling_eps = sampling_eps
        self.inverse_t = inverse_t
        self.loss_type = loss_type
        
        class_weight = torch.tensor([1.0] * vocab_size, device=device)
        class_weight[EOS_token] = eos_weight

        self.loss_fn = nn.CrossEntropyLoss(reduction='none', weight=class_weight)
        self.loss_fn = self.loss_fn.to(device)
        
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
        if self.loss_type == "eq8":
            logits = logits.clone()
            logits = self.subs_parameterisation(logits, xt)
        
            dt = 1 / self.T 
        
            timestep = timestep.clamp(self.sampling_eps, 1 - self.sampling_eps).view(-1, 1)

            #Extracts the logits corresponding to the true class labels
            log_prob_x = torch.gather(logits, -1, y_true[:, :, None]).squeeze(-1) 

            loss = - dt / timestep * log_prob_x
            loss = self.T * loss 

        elif self.loss_type == "eq9":
            logits = logits.clone()

            log_prob_mask = logits[:, :, MASK_token]
            prob_mask = log_prob_mask.exp()
            logits = self.subs_parameterisation(logits, xt)
        
            dt = 1 / self.T 
        
            timestep = timestep.clamp(dt + self.sampling_eps, 1 - self.sampling_eps).view(-1, 1)
        
            alpha_t = 1 - timestep + torch.zeros_like(xt)
            alpha_s = 1 - (timestep - dt) + torch.zeros_like(xt)
        
            log_prob_x = torch.gather(logits, -1, y_true[:, :, None]).squeeze(-1)
        
            term1_coef = dt / timestep 
            term1_num = torch.log(alpha_t * prob_mask / timestep + 1)
            term1_den = log_prob_x
        
            term2_coef = 1 - dt / timestep 
            term2_num = term1_num
            term2_den = torch.log(alpha_s * prob_mask / (timestep - dt) + 1)
        
            loss = term1_coef * (term1_num - term1_den) + term2_coef * (term2_num - term2_den)
        
            loss = self.T * loss * (xt == MASK_token)
        else:
            raise ValueError(f"{self.loss_type} is not defined.")

        # only calculate loss for non-padded tokens
        attention_mask = (y_true != PAD_token)
        loss = loss * attention_mask
        # calculate average loss
        loss = loss.sum() / attention_mask.sum()
        
        return loss


    # ELBO loss for T, (eq 9 in MDLM paper)
    def forward_old(self, xt, logits, y_true, timestep):
        logits = logits.clone()
        
        log_prob_mask = logits[:, :, MASK_token]
        prob_mask = log_prob_mask.exp()
        logits = self.subs_parameterisation(logits, xt)
        
        dt = 1 / self.T 
        
        timestep = timestep.clamp(dt + self.sampling_eps, 1 - self.sampling_eps).unsqueeze(-1)
        
        alpha_t = 1 - timestep + torch.zeros_like(xt)
        alpha_s = 1 - (timestep - dt) + torch.zeros_like(xt)
        
        log_prob_x = torch.gather(logits, -1, y_true[:, :, None]).squeeze(-1)
        
        term1_coef = dt / timestep 
        term1_num = torch.log(alpha_t * prob_mask / timestep + 1)
        term1_den = log_prob_x
        
        term2_coef = 1 - dt / timestep 
        term2_num = term1_num
        term2_den = torch.log(alpha_s * prob_mask / (timestep - dt) + 1)
        
        loss = term1_coef * (term1_num - term1_den) + term2_coef * (term2_num - term2_den)
        
        loss = self.T * loss * (xt == MASK_token)
        
        loss = loss.sum() / torch.numel(xt)
        #only calculate loss for non-padded tokens
        #attention_mask = (y_true != PAD_token)
        #loss = loss * attention_mask
        #calculate average loss
        #loss = loss.sum() / attention_mask.sum()
        
        return loss
    
    # lim (ELBO loss) as T -> inf
    # def forward(self, X, logits, y_true, timestep): 
    #     timestep = torch.clamp(timestep, min=0.01, max=1.0)  # No t < 0.01
    #     B, L = X.shape
    #     # X (B, L); float32
    #     # logits (B, L, 2); float32
    #     # y_true (B, L); long
        
    #     loss = self.loss_fn(logits.view(B*L, -1), y_true.view(B*L))
    #     loss = loss.view((B, L))
        
    #     mask = (X==MASK_token).float()
    #     loss *= mask

    #     # calculating weighted loss
    #     loss = loss.reshape((B, L))
        
    #     # if uniform sampling, apply 1/t wrighting
    #     if self.inverse_t == False:
    #         loss = 1.0/(timestep.unsqueeze(-1) + 1e-5) * loss
        
    #     loss = loss.sum() / B

    #     return loss
