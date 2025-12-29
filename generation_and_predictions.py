import torch 
import torch.nn as nn
import itertools

'''
Rule 1: #0s=#1s
Rule 2: substrings 010 and 101 are not allowed

'''

# Rule 1
def generate_seq(length):
    idxs = torch.arange(length).tolist()

    combs = list(itertools.combinations(idxs, length // 2))

    data = torch.zeros(len(combs), length).long()
    data[torch.arange(len(combs))[:, None], combs] = 1

    return data

# Rule 2
def select_rule_2(seqs):
    valid = []
    
    for seq in seqs:
        valid_seq = True 
        for idx in range(1, len(seq) - 1):
            if seq[idx] != seq[idx - 1] and seq[idx] != seq[idx + 1]:
                valid_seq = False 
                
        if valid_seq == True:
            valid.append(seq.unsqueeze(0))
    
    return torch.cat(valid, dim=0)    

def get_prediction_masked(model, input):
    seq = input.clone()
    
    model.eval()
    with torch.no_grad():
        while torch.sum((seq==2).long()) != 0:
            logits = model(seq)
            mask = (seq==2).nonzero()
            # prediction = torch.argmax(logits, dim=-1)
            # idx = torch.randint(0, mask.shape[0] - 1, (1,)).item() if mask.shape[0] > 1 else torch.tensor(0)
            # idx = mask[idx]
            
            norm_logits = torch.softmax(logits, dim=-1)
            vals = torch.maximum(norm_logits[:, :, 0], norm_logits[:, :, 1])
            vals[seq != 2] = float('-inf')
            idx = torch.argmax(vals)
            
            # seq[idx[0], idx[1]] = prediction[idx[0], idx[1]]
            seq[0, idx] = torch.argmax(logits, dim=-1)[0, idx]
                        
        return seq