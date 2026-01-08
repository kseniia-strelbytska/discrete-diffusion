import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
from generation_and_predictions import generate_seq, select_rule_2
from evaluation_tools import evaluation_loss, evaluation_from_generation

class Model(nn.Module):
    def __init__(self, max_len=20, vocab_size=3, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
        super().__init__() 

        self.n_head=n_head
        self.n_layers=n_layers
        self.embed_dim=embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X):
        B, L = X.shape 

        positions = self.positional_embedding(torch.arange(0, L).unsqueeze(0)) # (1, L) -> (1, L, E)
        X = self.embedding(X) + positions # (B, L, E)

        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()

        X  = self.transformer_encoder(src=X, mask=mask, is_causal=True) # apply mask to make it a unidirectional block!
        X = self.fc(X)

        return X

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.X.shape[0]
        
def train(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in tqdm(dataloader):
            B, L = X_batch.shape
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

    return model

def evaluation_loss(model, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        B, L = X_batch.shape
        with torch.no_grad():
            logits = model(X_batch)
            loss = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            total_loss += loss.item()
            # predictions = torch.argmax(logits, dim=-1)
            # print("Predictions:", predictions)
            # print("Ground Truth:", y_batch)
            
    print(f'Evaluation, Loss: {total_loss/len(dataloader)}')
  
if __name__ == '__main__':
    torch.manual_seed(47)
    l = 20
    
    seqs = generate_seq(l)
    seqs2 = select_rule_2(seqs)
    seqs, seqs2 = seqs2, seqs
    
    ### remove sequences starting with "0000"
    # spare = []
    # for seq in seqs:
    #     if seq[0:4].sum() == 0:
    #         continue 
        
    #     spare.append(seq.unsqueeze(0))
    
    # seqs = torch.cat(spare, dim=0)
    ###
    
    seqs = torch.cat([torch.full((seqs.shape[0], 1), torch.tensor(3)), seqs], dim=-1) # add start-of-sequence token
    
    X = seqs.clone()[:, :-1]
    y = seqs.clone()[:, 1:]
    
    dataset = Dataset(X, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]) 

    print(f'Dataset len: {len(dataset)}')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = Model(max_len=l, vocab_size=4, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1)
    model = train(model, train_dataloader, epochs=5, lr=1e-3)
    torch.save(model.state_dict(), f'./rule2_autoregressive_transformer')
    model.load_state_dict(torch.load('./rule2_autoregressive_transformer'))
        
    # evaluation_loss(model, test_dataloader)
    evaluation_from_generation(model, 20, 100, test_data)
    