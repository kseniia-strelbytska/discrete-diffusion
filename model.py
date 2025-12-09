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
        # Token embedding layer
        self.embedding_size = embedding_size
        self.l = l
        self.embed = nn.Embedding(vocab_size, embedding_size)
        
        self.time_encoding = nn.Linear(1, embedding_size)

         # Transformer/encoder layer
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=embedding_size, nhead=4, dim_feedforward=1024, dropout=0.1, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Predictor head: a simple linear layer
        self.predictor = nn.Linear(embedding_size, 2)

    def forward(self,
                input: torch.Tensor, timestep : torch.Tensor):
        # Turn input into embedding
        timestep = timestep.unsqueeze(1)

        embedded = self.embed(input.long()) # (B, L, E) = (128, 20, 10)
        B, L, E = embedded.shape

        position = torch.arange(0, L)[:, None].to(self.device) #Create a column vector [0,1,2,... max_len-1]^T representing token positions. Shape: [max_len,1]
        div_term = torch.exp(torch.arange(0, E, 2) / E * torch.log(torch.tensor([10000]))).to(self.device) #Define the division term seen in the formula. Defining 10000^... as a multiplication is numerically more stable
        pe = torch.zeros(1, L, E).to(self.device) #Creating a tensor of zeros for the positional encodings
        pe[0, :, 0::2] = torch.sin(position / div_term) #Selects every even indices along embedding dim
        pe[0, :, 1::2] = torch.cos(position / div_term) #Selects every odd indices along embedding dim
        embedded = embedded + pe

        time_pe = self.time_encoding(timestep)
        embedded = embedded + time_pe.unsqueeze(1)

        # Pass through network
        logits = self.encoder(embedded)

        # flatten b to (B*L, E), receive predictor result (B*L, 2), 
        # unflatten to (B, 2, L)
        logits = self.predictor(logits.view(-1, self.embedding_size)).view(-1, 2, self.l)

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

