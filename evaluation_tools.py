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

# eval_type: diffusion or autoregressive
# samples_type for anbn: random or full
def evaluation_from_generation(model, grammar, data=None, eval_type='diffusion', samples_type='random', n_samples=100):
    if data != None:
        data = data.clone()
    
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.5))))
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.75))))
    
    # r1, r2, both, format
    stats = np.array([0, 0, 0, 0])
    total = 0
    
    if eval_type == 'diffusion':
        if data == None:
            noise_level = 0.8
            data = grammar.data.clone()
            data[torch.rand_like(data, dtype=torch.float) < noise_level] = MASK_token
    else: 
        # test on prompts '000...0' and '000...01'
        data = grammar.data.clone()
        
        for l in range(1, grammar.l // 2 + 1):
            data[l - 1, l+2:] = MASK_token
            seq = data[l - 1].clone().unsqueeze(0)
            seq[:, l + 1] = MASK_token 
            data = torch.cat([data, seq], dim=0)
                    
    if samples_type == 'random':  
        data = data[torch.randperm(data.shape[0])]
        data = data[:n_samples]
                
    print(f'Evaluation on data, shape: f{data.shape}')                
    unmaskModel = ScheduledUnmasker(model)
    model.eval()
    with torch.no_grad():
        for s in tqdm(data):
            total += 1
            y_pred = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s))) # no batch dimension
            y_pred_stats = grammar.evaluate(y_pred)
            stats += y_pred_stats
            
    print(f'Evaluation from generation satisfies rule #1: {stats[0]}/{total} ({stats[0]/total})')
    print(f'Evaluation from generation satisfies rule #2: {stats[1]}/{total} ({stats[1]/total})')
    print(f'Evaluation from generation satisfies both rules: {stats[2]}/{total} ({stats[2]/total})')
    print(f'Evaluation from generation satisfies satisfies format: {stats[3]}/{total} ({stats[3]/total})')
    
    return stats / total
  