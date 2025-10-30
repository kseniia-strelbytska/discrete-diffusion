import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, data_loader, loss_fn, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        cs = []
        for X_batch, y_batch, alpha in data_loader:
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
            b = torch.argmax(a, dim=1)
            c = b.sum(dim=1)
            cs.extend(c.cpu())
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        sample = torch.tensor([[0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]], dtype = torch.long)
        out = model(sample)
        result = torch.argmax(out, dim=1)

        print(f"Epoch {epoch}, prediction: {result}")

        if epoch % 10 == 9:
            plt.hist(cs)
            plt.show()
    return model