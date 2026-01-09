import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from generation_and_predictions import get_prediction, get_prediction_fixedlen
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
                mask = (X_batch==2).view(B*L).float()
                loss *= mask
                loss = torch.sum(loss) / torch.sum(mask)
                
                total_loss += loss
                # predictions = torch.argmax(logits, dim=-1)
                # print("Predictions:", predictions)
                # print("Ground Truth:", y_batch)
            
    print(f'Evaluation, Loss: {total_loss/len(dataloader)}')

# eval_type: next_token or prefix
# samples_type for anbn: random or full
def evaluation_from_generation(model, grammar, data=None, eval_type='next_token', samples_type='random', n_samples=100):
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.5))))
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.75))))
    
    stats = np.array([0, 0, 0])
    total = 0
    if eval_type=='prefix': # used for initial grammar only; fixed length generation
        prefixes = grammar.data[:, :grammar.l//2]
        
        if data != None:
            prefixes = data[:, 1:grammar.l//2]
        
        idxs = torch.randint(0, prefixes.shape[0] - 1, (n_samples,))
        prefixes = prefixes[idxs]
           
        model.eval()
        with torch.no_grad():
            for prefix in prefixes:
                total += 1
                y_pred = get_prediction_fixedlen(model, prefix, grammar.l - prefix.shape[0] + 1)[1:] # no batch dim
                y_pred_stats = grammar.evaluate(y_pred)
                stats += y_pred_stats
    else:
        model.eval()
        
        if samples_type == 'random':
            samples = torch.randint(1, grammar.l//2, (n_samples,))
        else:
            samples = torch.arange(1, grammar.l//2)
        
        with torch.no_grad():
            for l in tqdm(samples.tolist()):
                seq = torch.cat([torch.full((1,), torch.tensor(SOS_token)), 
                                 torch.zeros((l, )).long()]) # has batch dim
                                
                # test on '000...0' and on '000...01'
                y_preds = [get_prediction(model, seq, grammar.l + 2), 
                        get_prediction(model, torch.cat([seq, torch.ones((1,)).long()], dim=-1), grammar.l + 2)] # +2 SOS/EOS    
                
                total += len(y_preds)
                
                for y_pred in y_preds:
                    y_pred_stats = grammar.evaluate(y_pred)
                    stats += y_pred_stats
                                                    
    print(f'Evaluation from generation satisfies rule #1: {stats[0]}/{total} ({stats[0]/total})')
    print(f'Evaluation from generation satisfies rule #2: {stats[1]}/{total} ({stats[1]/total})')
    print(f'Evaluation from generation satisfies both rules: {stats[2]}/{total} ({stats[2]/total})')
    
    return stats / total
  