import torch 
import torch.nn as nn
import itertools
from constants import EOS_token, SOS_token, PAD_token, MASK_token

# greedy decoding (greedily select position with the most confidence to unmask)
def get_prediction_masked(model, input):
    seq = input.clone()
    
    model.eval()
    with torch.no_grad():
        while torch.sum((seq==MASK_token).long()) != 0:
            logits = model(seq)
            mask = (seq==MASK_token).nonzero()
            # prediction = torch.argmax(logits, dim=-1)
            # idx = torch.randint(0, mask.shape[0] - 1, (1,)).item() if mask.shape[0] > 1 else torch.tensor(0)
            # idx = mask[idx]
            
            norm_logits = torch.softmax(logits, dim=-1)
            vals = torch.maximum(norm_logits[:, :, 0], norm_logits[:, :, 1])
            vals[seq != MASK_token] = float('-inf')
            idx = torch.argmax(vals)
            
            # seq[idx[0], idx[1]] = prediction[idx[0], idx[1]]
            seq[0, idx] = torch.argmax(logits, dim=-1)[0, idx]
                        
        return seq