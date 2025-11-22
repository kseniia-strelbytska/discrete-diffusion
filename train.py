import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from noise_schedule_unmask import get_scheduled_unmasker
from data_generation import sample_masked, generate_seq

def inference(model, data, figure_path='figures/'):
    # uniform masking (p_mask=0.5)
    unmaskModel = get_scheduled_unmasker(model, 0.1)

    valid = 0
    cs = []

    for idx, X in enumerate(tqdm(data)):
        y_pred = unmaskModel(X.unsqueeze(0))[0]

        if y_pred.sum() == y_pred.shape[0] // 2:
            valid += 1
        
        cs.append(y_pred.sum().to('cpu'))

    plt.hist(cs, bins=20, range = [0, 20])
    plt.xticks(range(0, 21))
    plt.savefig(f'./{figure_path}uniform_masking_inference')

    print(f'Valid solutions: {valid}/{data.shape[0]} ({valid/data.shape[0]})')

def train_model(model, data_loader, loss_fn, optimizer, device, num_epochs=50000, dict_path='models/', figure_path='figures/'):
    seqs = generate_seq(model.l)
    data = sample_masked(model.l, 100, torch.full((10**5, ), torch.tensor(0.5)), seqs)[:, 0, :] # (batch, 2, l) -> (batch, l)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        cs = []

        for X_batch, y_batch, alpha in tqdm(data_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            alpha = alpha.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch, alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            inference(model, data, figure_path)

            torch.save(model.state_dict(), f'./{dict_path}scaled_up_diffusion_model_{epoch + 1}epochs')
    
    return model