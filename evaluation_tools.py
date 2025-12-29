import torch
from tqdm import tqdm
from generation_and_predictions import generate_seq, get_prediction_masked
from noise_schedule_unmask import ScheduledUnmasker
from loss import rblb

def does_satisfy_rule1(seq):
    return (torch.sum(seq) == torch.numel(seq) // 2)

def does_satisfy_rule2(seq):
    for idx in range(1, len(seq) - 1):
        if seq[idx] != seq[idx - 1] and seq[idx] != seq[idx + 1]:
            return False 
    return True

def evaluation_loss(model, dataloader):
    loss_fn = rblb()
    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch, timestep in dataloader:
            B, L = X_batch.shape
            with torch.no_grad():
                logits = model(X_batch)
                loss = loss_fn(X_batch, logits, y_batch, timestep) 
                total_loss += loss
                # predictions = torch.argmax(logits, dim=-1)
                # print("Predictions:", predictions)
                # print("Ground Truth:", y_batch)
            
    print(f'Evaluation, Loss: {total_loss/len(dataloader)}')

def evaluation_from_generation(model, l, samples, data=None, epoch='nan', figure_path='figures/'):
    seqs = generate_seq(l) if data == None else data
    seqs = seqs[torch.randperm(seqs.shape[0])]
    seqs = seqs[:samples]
    
    unmaskModel = ScheduledUnmasker(model)

    total, printed, printed_incorrect = 0, 0, 0
    rule1, rule2, bothrules = 0, 0, 0 
    cs = []
    n_ones, n_masks = 0, 0
    
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(tqdm(seqs, desc="Evaluation from generation")):
            total += 1
            y_pred = unmaskModel(s, ((s == 2).sum() / torch.numel(s))) # no batch dimension
            
            # y_pred = get_prediction_masked(model, s.unsqueeze(0))
            
            r1, r2 = does_satisfy_rule1(y_pred), does_satisfy_rule2(y_pred)
            both = (r1 & r2)
            
            if r1 == True:
                rule1 += 1 
            if r2 == True:
                rule2 += 1 
            if both == True:
                bothrules += 1

            # if  both == True:
            #     if printed < 5:
            #         printed += 1
            #         print('Example of correct generative prediction: ', s.tolist(), y_pred.tolist())
            # else:
            #     if printed_incorrect < 5:
            #         printed_incorrect += 1
            #         print('Example of INcorrect generative prediction: ', s.tolist(), y_pred.tolist())

            # n_ones += y_pred[s==2].sum()
            # n_masks += (s==2).float().sum()
            # cs.append(y_pred.sum().to('cpu'))

        # plt.hist(cs, bins=model.l, range = [0, model.l])
        # plt.xticks(range(0, model.l + 1))
        # plt.savefig(f'./{figure_path}eval_mode_uniform_masking_inference_epoch{epoch}')

        print(f'Evaluation from generation satisfies rule #1: {rule1}/{total} ({rule1/total})')
        print(f'Evaluation from generation satisfies rule #2: {rule2}/{total} ({rule2/total})')
        print(f'Evaluation from generation satisfies btoh rules: {bothrules}/{total} ({bothrules/total})')
        
        return rule1, rule2, bothrules, total
        # print(f'Number of masks unmasked as 1s {n_ones} ({n_ones/n_masks})')