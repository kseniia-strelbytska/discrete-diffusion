
import torch
from constants import PAD_token, SOS_token, EOS_token
from torch import nn
import math
    
class ARTransformerClassifier(nn.Module):
    def __init__(self, 
                 max_len=20, 
                 vocab_size=5, 
                 n_head=4, 
                 n_layers=2, 
                 embed_dim=128, 
                 dim_feedforward=1024, 
                 layer_norm_eps=1e-5,
                 dropout=0.1,
                 sampling_eps=0):
        super().__init__() 
        self.architecture='autoregressive'

        self.n_head=n_head
        self.n_layers=n_layers
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.sampling_eps=sampling_eps  # not used; kept for interface compatibility
        
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                nhead=n_head, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout, 
                                                layer_norm_eps=layer_norm_eps,
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

        PE = torch.zeros((max_len, embed_dim))
        pos = torch.arange(max_len).unsqueeze(-1)
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2) / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("PE", PE)

    def forward(self, X, timestep=None):
        B, L = X.shape 
        mask = torch.triu(torch.full((L, L), float('-inf'), device=X.device), diagonal=1)
        padding_mask = (X == PAD_token).to(X.device) # (B, L)
    
        X = self.embedding(X) * math.sqrt(self.embed_dim) # (B, L, E)
        # Sinusoidal positional encoding
        X = self.dropout(X + self.PE[:L, :].unsqueeze(0)) # (B, L, E)

        X = self.transformer_encoder(src=X, 
                                     mask=mask, 
                                     src_key_padding_mask=padding_mask, 
                                     is_causal=True) # apply mask to make it a unidirectional block!
        X = self.fc(X)

        return X