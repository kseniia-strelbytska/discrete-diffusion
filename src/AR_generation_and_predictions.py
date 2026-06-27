import torch 
import torch.nn as nn
from datasets.constants import EOS_token, SOS_token, PAD_token, MASK_token


def _trim_prompt(seq):
    """Keep only the realised autoregressive prefix.

    Evaluation prompts are stored at full length and padded with `MASK_token`
    (and sometimes `PAD_token`). Autoregressive generation must ignore those
    placeholders and continue from the concrete prefix only.
    """
    stop_positions = []
    for stop_token in (MASK_token, PAD_token):
        positions = torch.where(seq == stop_token)[0]
        if len(positions) > 0:
            stop_positions.append(int(positions[0]))

    if stop_positions:
        seq = seq[: min(stop_positions)]

    # Safeguard if the prompt is already complete (i.e. contains EOS_token)
    eos_positions = torch.where(seq == EOS_token)[0]
    if len(eos_positions) > 0:
        seq = seq[: int(eos_positions[0]) + 1]

    return seq

def get_prediction_fixedlen(model, seq, extra_tokens): # no batch dim
    seq = _trim_prompt(seq)

    model.eval()
    with torch.no_grad():
        for token in range(extra_tokens):
            logits = model(seq.unsqueeze(0))[0] # no batch dim
            logits = logits[-1, :]
            prediction = torch.argmax(logits, dim=-1).unsqueeze(0)
            seq = torch.cat([seq, prediction], -1)
        return seq
    
def get_prediction(model, seq, max_tokens): # no batch dim
    seq = _trim_prompt(seq)

    if torch.numel(seq) > 0 and seq[-1].item() == EOS_token:
        return seq

    model.eval()
     
    with torch.no_grad():
        while torch.numel(seq) < max_tokens:
            logits = model(seq.unsqueeze(0))[0] # no batch dim
            logits = logits[-1, :]
            prediction = torch.argmax(logits, dim=-1).unsqueeze(0)
            seq = torch.cat([seq, prediction], -1)
            
            if prediction.item() == EOS_token:
                break
        return seq