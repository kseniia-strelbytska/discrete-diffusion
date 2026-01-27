import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from noise_schedule_unmask import ScheduledUnmasker
from constants import EOS_token, SOS_token, PAD_token, MASK_token

def evaluation_loss(model, dataloader, device='cpu'):
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_token).to(device)
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
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

def get_timeline(max_len=258, steps=None, idx=-1):
    ans = [(-1, -1)] * max_len 
    
    for t, step in enumerate(steps):
        for idx, val in enumerate(step):
            if val != 5:
                if ans[idx] == (-1, -1):
                    ans[idx] = (t, val.item())
    
    y_pred = steps[-1].clone()
    s, mid_start, pad_start = '', False, False
    s += "IDX " + str(idx) + " " + ''.join([str(i) for i in y_pred.tolist()]) + '\n'
    cnt_zeros, cnt_ones = (y_pred==0).sum(), (y_pred==1).sum()
    s += f' zeros={cnt_zeros}, ones={cnt_ones} \n'
    
    for idx, i in enumerate(ans):
        s += f'{idx:>4}: {i[1]:<20} set at {i[0]:>10}\n'
        if idx + 1 < len(ans) and ans[idx + 1][1] == 1 and mid_start == False:
            s += '------MIDDLE------\n'
            mid_start = True
        elif idx + 1 < len(ans) and ans[idx + 1][1] == PAD_token and pad_start == False:
            s += '------START_OF_PADDING------\n'
            pad_start = True
    
    return ans, s

# eval_type: diffusion or autoregressive
# samples_type for anbn: random or full
def evaluation_from_generation(model, grammar, data=None, T=500, eval_type='diffusion', samples_type='random', n_samples=100, write_steps=False, device='cpu', figures_path=None, loss_log_path=None, output_path=None):
    if data != None:
        data = data.clone()

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

    data = data.to(device)
    
    with open(output_path, 'w') as f:
        f.write('')
    
    print(f'Evaluation on data, shape: f{data.shape}')                
    unmaskModel = ScheduledUnmasker(model, T=T, device=device)
    
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(tqdm(data)):
            total += 1
            if write_steps == False:
                y_pred = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s))) # no batch dimension
            else:
                y_pred, steps = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s)), return_steps=True)
            y_pred_stats = grammar.evaluate(y_pred)
            stats += y_pred_stats
            
            with open(output_path, 'a') as f:
                f.write("IDX " + str(idx) + " " + ''.join([str(i) for i in y_pred.tolist()]))
                is_format_ok = ('True' if y_pred_stats[-1] == 1 else 'False')
                cnt_zeros, cnt_ones = (y_pred==0).sum(), (y_pred==1).sum()
                f.write(f' zeros={cnt_zeros}, ones={cnt_ones}, format={is_format_ok} \n')
                
                if write_steps == True:
                    f.write('Full denoising log: \n')
                    prev = torch.tensor([0])
                    for step in steps:
                        if step.tolist() != prev.tolist():
                            f.write(''.join([str(i) for i in step.tolist()]) + '\n')
                            cnt_zeros, cnt_ones, masks = (step==0).sum(), (step==1).sum(), (step==MASK_token).sum()
                            f.write(f' zeros={cnt_zeros}, ones={cnt_ones}, masks={masks} \n')
                            prev = step
                        
                    f.write('-' * 30 + '\n')
                    
            if write_steps == True:
                with open(figures_path / f'IDX={idx}.txt', 'a') as f:
                    ans, output_str = get_timeline(max_len=grammar.l+2, steps=steps, idx=idx)  
                    f.write(output_str)
    
    evaluation_log = f"""
    Evaluation from generation satisfies rule #1: {stats[0]}/{total} ({stats[0]/total})
    Evaluation from generation satisfies rule #2: {stats[1]}/{total} ({stats[1]/total})
    Evaluation from generation satisfies both rules: {stats[2]}/{total} ({stats[2]/total})
    Evaluation from generation satisfies satisfies format: {stats[3]}/{total} ({stats[3]/total})
    """
    with open(loss_log_path, 'a') as f:
        f.write(evaluation_log+'\n')            
    
    print(evaluation_log)
    
    return stats / total
  
