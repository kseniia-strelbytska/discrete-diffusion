import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from noise_schedule_unmask import SequencedScheduledUnmasker
from data_generation import sample_masked, generate_seq
from loss import rblb

# def train_model(model, data_loader, epochs=5, lr=1e-3, device='cpu', dict_path='models/', figure_path='figures/'):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     loss_fn = rblb(device=device)
    
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         cs = []
        
#         for X_batch, y_batch, timestep in tqdm(data_loader, desc=f"Training epoch #{epoch + 1}"):
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)
#             timestep = timestep.to(device)

#             optimizer.zero_grad()
#             y_pred = model(X_batch)
#             loss = loss_fn(X_batch, y_pred, y_batch, timestep)
#             loss.backward()

#             optimizer.step()
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(data_loader)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#         if (epoch + 1) % 1 == 0:
#             torch.save(model.state_dict(), f'./{dict_path}scaled_up_diffusion_model_{epoch + 1}epochs')
    
#     return model