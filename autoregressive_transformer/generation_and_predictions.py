import torch 
import torch.nn as nn
import itertools

def generate_seq(length):
    idxs = torch.arange(length).tolist()

    combs = list(itertools.combinations(idxs, length // 2))

    data = torch.zeros(len(combs), length).long()
    data[torch.arange(len(combs))[:, None], combs] = 1

    return data

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

def mask_seq(seqs, probs):
    # our goal is to mask each seq_i with p(seq_i_j=mask) = probs_i 
    
    mask = torch.rand_like(seqs, dtype = torch.float) 
    mask = mask < probs.unsqueeze(-1)
    masked_seqs = seqs.clone()
    masked_seqs = torch.where(mask==True, torch.full_like(seqs, torch.tensor(2)), seqs)
    
    return masked_seqs

def get_prediction(model, seq, extra_tokens):
    model.eval()
    with torch.no_grad():
        for token in range(extra_tokens):
            logits = model(seq)
            logits = logits[0, -1, :]
            prediction = torch.tensor([torch.argmax(logits, dim=-1)]).unsqueeze(0)
            seq = torch.cat([seq, prediction], -1)
        return seq