import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from noise_schedule_unmask import ScheduledUnmasker
from constants import EOS_token, SOS_token, PAD_token, MASK_token

def evaluation_loss(model, dataloader):
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_token)
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            B, L = X_batch.shape
            with torch.no_grad():
                logits = model(X_batch)
                loss = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
                mask = (X_batch==MASK_token).view(B*L).float()
                loss *= mask
                loss = torch.sum(loss) / torch.sum(mask)
                
                total_loss += loss
                # predictions = torch.argmax(logits, dim=-1)
                # print("Predictions:", predictions)
                # print("Ground Truth:", y_batch)
            
    print(f'Evaluation, Loss: {total_loss/len(dataloader)}')

# eval_type: next_token or prefix
# samples_type for anbn: random or full
def evaluation_from_generation(model, grammar, data=None, samples_type='random', n_samples=100):
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.5))))
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.75))))
    
    stats = np.array([0, 0, 0])
    total = 0
    
    if data == None:
        noise_level = 0.8
        data = grammar.data.clone()
        data[torch.rand_like(data, dtype=torch.float) < noise_level] = MASK_token
    
    data = data[torch.randperm(data.shape[0])]
    data = data[:n_samples]
        
    unmaskModel = ScheduledUnmasker(model)

    model.eval()
    with torch.no_grad():
        for s in data:
            total += 1
            y_pred = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s))) # no batch dimension
            y_pred_stats = grammar.evaluate(y_pred)
            stats += y_pred_stats
              
    print(f'Evaluation from generation satisfies rule #1: {stats[0]}/{total} ({stats[0]/total})')
    print(f'Evaluation from generation satisfies rule #2: {stats[1]}/{total} ({stats[1]/total})')
    print(f'Evaluation from generation satisfies both rules: {stats[2]}/{total} ({stats[2]/total})')
    
    return stats / total
  