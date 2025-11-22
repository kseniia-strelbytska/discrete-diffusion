import torch
import torch.nn as nn

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=5, dim_feedforward=1024, dropout=0.1, layer_norm_eps=6e-3)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        # Predictor head: a simple linear layer
        self.l = l
        self.predictor = nn.Linear(embedding_size, 2)

    def forward(self,
                input: torch.Tensor):
        # Turn input into embedding
        embedded = self.embed(input.long())
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

