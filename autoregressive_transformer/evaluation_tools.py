import torch 
import torch.nn as nn
from tqdm import tqdm
from generation_and_predictions import generate_seq, get_prediction, select_rule_2

def does_satisfy_rule1(seq):
    return (torch.sum(seq) == torch.numel(seq) // 2)

def does_satisfy_rule2(seq):
    for idx in range(1, len(seq) - 1):
        if seq[idx] != seq[idx - 1] and seq[idx] != seq[idx + 1]:
            return False 
    return True

def evaluation_loss(model, dataloader):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
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

def evaluation_from_generation(model, l, samples, data=None):
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.5))))
    # prefixes = select_rule_2(generate_seq(max(1, round(l * 0.75))))
    prefixes = torch.full((samples, 4), torch.tensor(0)) 
    
    if data != None:
        prefixes = data[:, 1:round(0.2 * l)]
    
    idxs = torch.randint(0, prefixes.shape[0] - 1, (samples,))
    prefixes = prefixes[idxs]

    prefixes = torch.cat([torch.full((prefixes.shape[0], 1), torch.tensor(3)), prefixes], dim=-1)
    
    model.eval()
    rule1, rule2, bothrules, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for prefix in prefixes:
            total += 1
            y_pred = get_prediction(model, prefix.unsqueeze(0), l - prefix.shape[0] + 1)[:, 1:][0]
            
            r1, r2 = does_satisfy_rule1(y_pred), does_satisfy_rule2(y_pred)
            both = (r1 & r2)
            
            if r1 == True:
                rule1 += 1 
            if r2 == True:
                rule2 += 1 
            if both == True:
                bothrules += 1
                print(y_pred)
                
            print(y_pred)
                
    print(f'Evaluation from generation satisfies rule #1: {rule1}/{total} ({rule1/total})')
    print(f'Evaluation from generation satisfies rule #2: {rule2}/{total} ({rule2/total})')
    print(f'Evaluation from generation satisfies both rules: {bothrules}/{total} ({bothrules/total})')
  