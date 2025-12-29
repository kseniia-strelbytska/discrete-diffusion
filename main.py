import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from evaluation_tools import evaluation_loss, evaluation_from_generation
from loss import rblb
from generation_and_predictions import generate_seq, select_rule_2
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
        
        PE = torch.zeros((max_len, embed_dim))
        pos = torch.arange(max_len).unsqueeze(-1)
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2) / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)
        
        self.register_buffer('PE', PE)

    def forward(self,
                X: torch.Tensor):
        B, L = X.shape
        X = self.embedding(X) # (B, L, E) = (128, 20, 10)     
        E = X.shape[-1]
        
        ## Sinusoidal positional encoding 
        X += self.PE[:L, :].unsqueeze(0)

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

def train(model, dataloader, epochs=5, lr=1e-3, dict_path='models/', figure_path='figures/', test_data = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = rblb()
    
    rule1_data, rule2_data, bothrules_data, total_data = [], [], [], []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        cs = []
        
        model.train()
        for X_batch, y_batch, timestep in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(X_batch, logits, y_batch, timestep)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        rule1, rule2, bothrules, total = evaluation_from_generation(model, model.l, 100, test_data, 0)
        rule1_data.append(rule1/total)
        rule2_data.append(rule2/total)
        bothrules_data.append(bothrules/total)

        epochtimes = 1000 + np.arange(epoch + 1)
        plt.plot(epochtimes, rule1_data)
        plt.plot(epochtimes, rule2_data)
        plt.plot(epochtimes, bothrules_data)
        plt.legend(["Rule 1", "Rule 2", "Both Rules"], loc="lower right")
        plt.savefig('./plot')
        plt.clf()

    return model

if __name__ == '__main__':
    l = 16

    seqs = generate_seq(l)
    seqs2 = select_rule_2(seqs)
    seqs, seqs2 = seqs2, seqs    
        
    dataset = Dataset(seqs)        
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print(f'Dataset len: {len(dataset)}')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]) 
    hardcore_data = seqs.clone()
    p = 0.8 + 0.2 * torch.rand((seqs.shape[0], 1))
    mask = torch.rand_like(hardcore_data, dtype=torch.float) < p
    hardcore_data[mask] = 2    
    
    model = TransformerClassifier(max_len=l, vocab_size=4, n_head=4, n_layers=4, embed_dim=64, dim_feedforward=128, dropout=0.1)
    model.load_state_dict(torch.load('./models/diffusion_transformer_1000_epochs_embd_64'))
    model = train(model=model, dataloader=train_dataloader, epochs=1000, lr=1e-3, dict_path='models/test/', figure_path='figures/test/', test_data=hardcore_data)
    torch.save(model.state_dict(), f'./models/diffusion_transformer_2000_epochs_embd_64')
    
    unmask = ScheduledUnmasker(model)
    X = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    y = unmask(X, (torch.sum(X) / torch.numel(X)))
    print(y.sum())
    print(y.tolist())
        
    evaluation_loss(model, test_dataloader)
    evaluation_from_generation(model, l, 100, data=test_data)
    evaluation_from_generation(model, l, 1000, data=hardcore_data)
