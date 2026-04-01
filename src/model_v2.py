import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class v2MultiheadAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        layer_norm_eps=2e-4,
        batch_first=True,
        norm_first=False,
    ):
        super().__init__()
        self.nhead = nhead
        self.norm_first = norm_first
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        
        self.proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        
        self.relu = nn.ReLU()
        self.norm_attn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_ff = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout_actv = nn.Dropout(dropout)
        self.dropout_ff1 = nn.Dropout(dropout)
        self.dropout_ff2 = nn.Dropout(dropout)

    
    def _sa_block(self, X):
        qkv = self.qkv(X)
        qkv = qkv.reshape(X.shape[0], X.shape[1], 3, self.nhead, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, nhead, L, d_model/nhead)
        Q, K, V = qkv.unbind(0) # (B, nhead, L, d_model/nhead)
        
        attn = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]) # (B, nhead, L, L)
        attn = F.softmax(attn, dim=-1) # (B, nhead, L, L)
        
        out = attn @ V # (B, nhead, L, d_model/nhead)
        out = out.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[1], -1) # (B, L, d_model)  
        out = self.dropout_actv(self.proj(out))
        
        return out
    
    def _ff_block(self, X):
        X = self.dropout_ff2(self.ff2(self.dropout_ff1(self.relu(self.ff1(X)))))
        
        return X
    
    def forward(self, X):
        if self.norm_first:
            X = X + self._sa_block(self.norm_attn(X))
            X = X + self._ff_block(self.norm_ff(X))
        else:
            X = self.norm_attn(X + self._sa_block(X))
            X = self.norm_ff(X + self._ff_block(X))
        
        return X

class v2TransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        max_len=16,
        vocab_size=6,
        n_head=4,
        n_layers=2,
        embed_dim=128,
        dim_feedforward=1024,
        dropout=0.1,
        layer_norm_eps=2e-4,
        sampling_eps=1e-5,
    ):
        super().__init__()
        self.architecture='diffusion'

        self.l = max_len
        self.sampling_eps = sampling_eps
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.sigma_map = TimestepEmbedder(embed_dim=embed_dim)

        self.transformer_encoder = nn.Sequential(
            *[v2MultiheadAttentionLayer(d_model=embed_dim, 
                                      nhead=n_head, 
                                      dim_feedforward=dim_feedforward,
                                      dropout=dropout, 
                                      layer_norm_eps=layer_norm_eps,
                                      batch_first=True,
                                      norm_first=False) for _ in range(n_layers)])  

        # Predictor head: a simple linear layer
        # do not allow mask (5) prediction
        self.fc = nn.Linear(embed_dim, vocab_size)

        PE = torch.zeros((max_len, embed_dim))
        pos = torch.arange(max_len).unsqueeze(-1)
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2) / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("PE", PE)

    def forward(self, X: torch.Tensor, timestep: torch.Tensor):
        B, L = X.shape
        X = self.embedding(X)  # (B, L, E) = (128, 20, 10)
        E = X.shape[-1]

        # Sinusoidal positional encoding
        X += self.PE[:L, :].unsqueeze(0)
        
        # Sinusoidal timestep encoding
        #c = torch.nn.functional.silu(self.sigma_map(-torch.log(1 - (1 - self.sampling_eps) * timestep)))
        #X += c

        # Pass through network
        X = self.transformer_encoder(X)
        X = self.fc(X)

        return X
