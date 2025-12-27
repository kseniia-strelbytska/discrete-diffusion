import torch
from tqdm import tqdm
from generation_and_predictions import generate_seq, get_prediction_masked
from loss import rblb

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
    
    # unmaskModel = SequencedScheduledUnmasker(model, 0.02).to(model.device)

    correct, total, printed = 0, 0, 0
    cs = []
    n_ones, n_masks = 0, 0
    
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(tqdm(seqs, desc="Inference")):
            total += 1
            
            p = torch.rand((1, ))
            mask = torch.rand_like(s, dtype=torch.float) < p.item()
            s[mask == True] = 2
            # y_pred = unmaskModel(s.unsqueeze(0), p.unsqueeze(0)).to(model.device)[0] # remove batch dimension
            
            y_pred = get_prediction_masked(model, s.unsqueeze(0))

            if torch.sum(y_pred) == l // 2:
                correct += 1
                
                if printed < 5:
                    printed += 1
                    print('Example of correct generative prediction: ', p.item(), s.tolist(), y_pred.tolist())

            # n_ones += y_pred[s==2].sum()
            # n_masks += (s==2).float().sum()
            # cs.append(y_pred.sum().to('cpu'))

        # plt.hist(cs, bins=model.l, range = [0, model.l])
        # plt.xticks(range(0, model.l + 1))
        # plt.savefig(f'./{figure_path}eval_mode_uniform_masking_inference_epoch{epoch}')

        print(f'Evaluation from generation: {correct}/{total} ({correct/total})')
        # print(f'Number of masks unmasked as 1s {n_ones} ({n_ones/n_masks})')