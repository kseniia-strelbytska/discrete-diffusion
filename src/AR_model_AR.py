
import torch
from constants import PAD_token, SOS_token, EOS_token
from torch import nn
    
class ARTransformerClassifier(nn.Module):
    def __init__(self, 
                 max_len=20, 
                 vocab_size=5, 
                 n_head=4, 
                 n_layers=2, 
                 embed_dim=128, 
                 dim_feedforward=1024, 
                 dropout=0.1,
                 sampling_eps=0):
        super().__init__() 
        self.architecture='autoregressive'

        self.n_head=n_head
        self.n_layers=n_layers
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.sampling_eps=sampling_eps  # not used; kept for interface compatibility

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X, timestep=None):
        B, L = X.shape 
        mask = torch.triu(torch.ones(L, L, device=X.device), diagonal=1).bool()
        padding_mask = (X == PAD_token)
    
        positions = self.positional_embedding(torch.arange(0, L, device=X.device).unsqueeze(0)) # (1, L) -> (1, L, E)
        X = self.embedding(X) + positions # (B, L, E)

        X = self.transformer_encoder(src=X, mask=mask, src_key_padding_mask=padding_mask, is_causal=True) # apply mask to make it a unidirectional block!
        X = self.fc(X)

        return X