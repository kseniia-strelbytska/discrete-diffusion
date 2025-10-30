import torch
import torch.nn as nn

class TransformerClassifier(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_layers: int = 2,
                 embedding_size: int = 8,
                 l: int = 20):
        super().__init__()
        # Embedding layer
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # Transformer/encoder layer
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, 2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Predictor head: a simple linear layer
        self.l = l
        self.predictor = nn.Linear(embedding_size, 2)

    def forward(self,
                input: torch.Tensor):
        # Turn input into embedding
        embedded = self.embed(input.to(torch.long))
        # Pass through network
        logits = self.forward_with_embedding(embedded)
        return logits

    def forward_with_embedding(self, embedded: torch.Tensor):
        b = self.encoder(embedded)
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

