import torch
import torch.nn as nn
import math

class TransformerClassifier(torch.nn.Module):
    def __init__(self,
                 device: 'cpu',
                 vocab_size: int,
                 num_layers: int = 7,
                 embedding_size: int = 10,
                 l: int = 20):
        super().__init__()

        self.device = device
        # Embedding layer
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)

        # Transformer/encoder layer
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=embedding_size, nhead=5, dim_feedforward=1024, dropout=0.1, layer_norm_eps=6e-3)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        # Predictor head: a simple linear layer
        self.l = l
        self.predictor = nn.Linear(embedding_size, 2)

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self,
                input: torch.Tensor, timestep : torch.Tensor):
        # Turn input into embedding
        embedded = self.embed(input.long()) # (B, L, E) = (128, 20, 10)
        timestep = timestep.unsqueeze(1).unsqueeze(2) * torch.tensor(1000) # (128, 1, 1)

        B, L, E = embedded.shape

        div_term = torch.exp(torch.arange(0, E, 2) / E * torch.log(torch.tensor([10000]))).to(self.device) #Define the division term seen in the formula. Defining 10000^... as a multiplication is numerically more stable
        div_term = div_term.unsqueeze(0).unsqueeze(0) # (1, 1, E/2)

        pe = torch.zeros(B, L, E).to(self.device) # (128, 20, 10)
        pe[:, :, 0::2] = torch.sin(timestep / div_term) # (128, 20, 5) = (128, 1, 5)
        pe[:, :, 1::2] = torch.cos(timestep / div_term) # (128, 20, 5) = (128, 1, 5)

        pe = self.time_mlp(pe)

        embedded = embedded + pe

        # Pass through network
        logits = self.forward_with_embedding(embedded)
        return logits

    def forward_with_embedding(self, embedded: torch.Tensor):
        # (B, L, E) = (batch, sequence length, embedding size)
        b = self.encoder(embedded) # (B, L, E)
        # flatten b to (B*L, E), receive predictor result (B*L, 2), 
        # unflatten to (B, 2, L)
        logits = self.predictor(b.view(-1, self.embedding_size)).view(-1, 2, self.l)

        return logits

class Model(torch.nn.Module):
    def __init__(self, dim, category_count, hidden_count1, hidden_count2):
        super().__init__()
        self.dim = dim
        self.category_count = category_count
        self.l1 = torch.nn.Linear(dim, hidden_count1)
        self.l2 = torch.nn.Linear(hidden_count1, hidden_count2)
        self.l3 = torch.nn.Linear(hidden_count2, dim*category_count)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        X = self.l1(X)
        X = self.relu(X)
        X = self.l2(X)
        X = self.relu(X)
        X = self.l3(X)

        return X.view(-1, self.category_count, self.dim)

