import torch
import torch.nn as nn
import math

# class TransformerClassifier(torch.nn.Module):
#     def __init__(self, device='cpu',
#                  max_len=16, vocab_size=3, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
#         super().__init__()

#         self.device = device
#         # Token embedding layer
#         self.embedding_size = embed_dim
#         self.l = max_len
        
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#          # Transformer/encoder layer
#         self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

#         # Predictor head: a simple linear layer
#         self.fc = nn.Linear(embed_dim, 2)

#     def forward(self,
#                 input: torch.Tensor):
#         B, L = input.shape
#         embedded = self.embedding(input.to(self.device)) # (B, L, E) = (128, 20, 10)        
#         # Pass through network
#         logits = self.transformer_encoder(embedded)
#         logits = self.fc(logits)

#         return logits