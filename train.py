import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from noise_schedule_unmask import SequencedScheduledUnmasker
from data_generation import sample_masked, generate_seq

def inference(model, data, epoch, figure_path='figures/'):
    model.eval()
    # uniform masking (p_mask=0.5)
    # unmaskModel = get_scheduled_unmasker(model, 0.02)
    unmaskModel = SequencedScheduledUnmasker(model, 0.02).to(model.device)

    valid = 0
    cs = []
    n_ones, n_masks = 0, 0

    for idx, X in enumerate(tqdm(data, desc="Inference")):
        # X shape: (20)

        timestep = torch.full((1,), torch.tensor(0.5)).to(model.device)
        y_pred = unmaskModel(X.unsqueeze(0), timestep).to(model.device)[0]

        # y_pred shape: (2)

        n_ones += y_pred[X==2].sum()
        n_masks += y_pred[X==2].shape[0]

        if y_pred.sum() == y_pred.shape[0] // 2:
            valid += 1
        
        cs.append(y_pred.sum().to('cpu'))

    plt.hist(cs, bins=model.l, range = [0, model.l])
    plt.xticks(range(0, model.l + 1))
    plt.savefig(f'./{figure_path}eval_mode_uniform_masking_inference_epoch{epoch}')

    print(f'Valid solutions: {valid}/{data.shape[0]} ({valid/data.shape[0]})')
    print(f'Number of masks unmasked as 1s {n_ones} ({n_ones/n_masks})')

    model.train()

def train_model(model, data_loader, loss_fn, optimizer, device, num_epochs=50000, dict_path='models/', figure_path='figures/'):
    seqs = generate_seq(model.l)
    data = sample_masked(model.l, 100, torch.full((10**5, ), torch.tensor(0.5)), seqs)[:, 0, :] # (batch, 2, l) -> (batch, l)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        cs = []

        for X_batch, y_batch, timestep in tqdm(data_loader, desc=f"Training epoch #{epoch + 1}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            timestep = timestep.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch, timestep)
            loss = loss_fn(X_batch, y_pred, y_batch, timestep)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 1 == 0:
            inference(model, data, epoch + 1, figure_path)

            torch.save(model.state_dict(), f'./{dict_path}scaled_up_diffusion_model_{epoch + 1}epochs')
    
    return model