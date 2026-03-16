import torch
import torch.nn as nn
import math



class TimestepEmbedder(torch.nn.Module):
    '''
    Takes in (B, 1): timestep=-log(1-t) (possibly fractional)
    Returns: (B, embed_dim): vector representation
    '''
    def __init__(self, embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
        - math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
            [embedding,
            torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
         
        return t_emb

class TransformerClassifier(torch.nn.Module):
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

        self.l = max_len
        self.sampling_eps = sampling_eps
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sigma_map = TimestepEmbedder(embed_dim=embed_dim)

        # Transformer/encoder layer
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=n_layers
        )

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
        t = timestep.view(-1)  # ensure 1D shape (B,)
        sigma = -torch.log(1 - (1 - self.sampling_eps) * t)
        c = self.sigma_map(sigma)  # (B, embed_dim)
        X = X + c.unsqueeze(1)  # broadcast to (B, L, embed_dim)

        # Pass through network
        X = self.transformer_encoder(src=X)
        X = self.fc(X)

        return X
