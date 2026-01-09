import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from evaluation_tools import evaluation_loss, evaluation_from_generation
from generation_and_predictions import get_prediction
from anbn import anbnGrammar
from initialgrammar import initialGrammar
from constants import EOS_token, SOS_token, PAD_token, MASK_token

class Model(nn.Module):
    def __init__(self, max_len=20, vocab_size=6, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
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
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        padding_mask = (X == PAD_token)
    
        positions = self.positional_embedding(torch.arange(0, L).unsqueeze(0)) # (1, L) -> (1, L, E)
        X = self.embedding(X) + positions # (B, L, E)

        X  = self.transformer_encoder(src=X, mask=mask, src_key_padding_mask=padding_mask, is_causal=True) # apply mask to make it a unidirectional block!
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
    class_weights = torch.ones(6)
    class_weights[EOS_token] = 10.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token, weight=class_weights)
    
    stats = [[], [], [], []] #r1, r2, both, epochsteps

    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        for X_batch, y_batch in tqdm(dataloader):
            B, L = X_batch.shape
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            new_stats = evaluation_from_generation(model, grammar, data=None, eval_type=grammar.default_eval_type, samples_type='full', n_samples=100)
            
            for i in range(3):
                stats[i].append(new_stats[i]) 
            stats[3].append(epoch + 1)
            
            plt.plot(stats[3], stats[0])
            plt.plot(stats[3], stats[1])
            plt.plot(stats[3], stats[2])
            plt.legend(["Rule 1", "Rule 2", "Both Rules"], loc="lower right")
            plt.savefig('./plot')
            plt.clf()
            
            torch.save(model.state_dict(), f'./models/initialgrammar_autoregressive_transformer_epochs={epoch + 1}')
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

    return model

if __name__ == '__main__':
    torch.manual_seed(1)
    l = 256
    
    grammar = anbnGrammar(l)
    grammar.data = grammar.generate_seq()
    
    X = grammar.data.clone()[:, :-1]
    y = grammar.data.clone()[:, 1:]
    
    dataset = Dataset(X, y)
    print(f'Dataset len: {len(dataset)}')
    
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]) 
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    full_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model(max_len=l+2, vocab_size=6, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1) # +2 SOS/EOS token
    model.load_state_dict(torch.load('./models/anbn_trained_models/rule2_autoregressive_transformer_epochs=1500'))
    # model = train(model, full_dataloader, epochs=2000, lr=1e-3)
    # torch.save(model.state_dict(), f'./rule2_autoregressive_transformer_500')
                
    # evaluation_loss(model, test_dataloader)
    evaluation_from_generation(model, grammar, data=None, eval_type='next_token', samples_type='full', n_samples=20)
    