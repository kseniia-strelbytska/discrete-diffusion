import torch
import torch.nn as nn
from tqdm import tqdm
from evaluation_tools import evaluation_loss, evaluation_from_generation
from loss import rblb
from generation_and_predictions import generate_seq
from noise_schedule_unmask import ScheduledUnmasker

class TransformerClassifier(torch.nn.Module):
    def __init__(self, max_len=16, vocab_size=3, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.l = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
         # Transformer/encoder layer
        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        # Predictor head: a simple linear layer
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self,
                X: torch.Tensor):
        B, L = X.shape
        X = self.embedding(X) # (B, L, E) = (128, 20, 10)        
        # Pass through network
        X = self.transformer_encoder(src=X)
        X = self.fc(X)

        return X

class Dataset(torch.utils.data.Dataset):
    def __init__(self, y):
        self.y = y
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        y_sample = self.y[index]
        prob = torch.rand((1, )) # prob of having a mask (ie the timestep)
        mask = torch.rand_like(y_sample, dtype=torch.float) < prob.item()
        X_sample = torch.where(mask == True, torch.full_like(y_sample, torch.tensor(2)), y_sample)
        
        return X_sample, y_sample, prob

def train(model, dataloader, epochs=5, lr=1e-3, dict_path='models/', figure_path='figures/'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = rblb()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        cs = []
        
        for X_batch, y_batch, timestep in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(X_batch, logits, y_batch, timestep)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # if (epoch + 1) % 1 == 0:
        #     torch.save(model.state_dict(), f'./{dict_path}scaled_up_diffusion_model_{epoch + 1}epochs')
    
    return model

if __name__ == '__main__':
    l = 16

    seqs = generate_seq(l)
    dataset = Dataset(seqs)        
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(f'Dataset len: {len(dataset)}')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    model = TransformerClassifier(max_len=l, vocab_size=4, n_head=4, n_layers=4, embed_dim=128, dim_feedforward=128, dropout=0.1)
    # model = train(model=model, dataloader=train_dataloader, epochs=6, lr=1e-3, dict_path='models/test/', figure_path='figures/test/')
    # torch.save(model.state_dict(), f'./diffusion_transformer')
    model.load_state_dict(torch.load('./diffusion_transformer'))

    evaluation_loss(model, test_dataloader)
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])    
    evaluation_from_generation(model, l, 10, data=test_data)
    
    hardcore_data = seqs.clone()
    p = 0.8 + 0.2 * torch.rand((seqs.shape[0], 1))
    mask = torch.rand_like(hardcore_data, dtype=torch.float) < p
    hardcore_data[mask] = 2     
    evaluation_from_generation(model, l, 100, data=hardcore_data)
