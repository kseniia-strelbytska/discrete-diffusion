import torch 
import torch.nn as nn
from constants import EOS_token, SOS_token, PAD_token, MASK_token

def get_prediction_fixedlen(model, seq, extra_tokens): # no batch dim
    model.eval()
    with torch.no_grad():
        for token in range(extra_tokens):
            logits = model(seq.unsqueeze(0))[0] # no batch dim
            logits = logits[-1, :]
            prediction = torch.tensor([torch.argmax(logits, dim=-1)])
            seq = torch.cat([seq, prediction], -1)
        return seq
    
def get_prediction(model, seq, max_tokens): # no batch dim
    model.eval()
     
    with torch.no_grad():
        while torch.numel(seq) < max_tokens:
            logits = model(seq.unsqueeze(0))[0] # no batch dim
            logits = logits[-1, :]
            prediction = torch.tensor([torch.argmax(logits, dim=-1)])
            seq = torch.cat([seq, prediction], -1)
            
            if prediction.item() == EOS_token:
                break
        return seq