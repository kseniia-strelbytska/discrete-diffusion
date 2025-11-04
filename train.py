import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, data_loader, loss_fn, optimizer, device, num_epochs=10):
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
            a = nn.functional.softmax(y_pred, dim=-1)
            b = (X_batch == 2) * torch.argmax(a, dim=1) + (X_batch != 2) * X_batch
            c = b.sum(dim=1)
            cs.extend(c.cpu())
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        sample = torch.tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = torch.float).to(device)
        # sample = torch.tensor([[0, 2, 0, 0, 1, 1]], dtype = torch.float)

        out = model(sample)
        result = torch.argmax(out, dim=1)

        print(f"Epoch {epoch}, prediction: {result}")

        if epoch % 10 == 0:
            plt.hist(cs, bins = model.l, range = [0.0, model.l])
            plt.xticks(range(0, model.l + 1))
            plt.savefig(f'./figures/inverse_t_{epoch + 1 + 30}epochs')

            torch.save(model.state_dict(), f'./models/diffusion_model_31_10_{epoch + 1 + 30}epochs')
    
    return model