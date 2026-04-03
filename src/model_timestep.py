import torch
import torch.nn as nn
import math

from constants import PAD_token, SOS_token, EOS_token, MASK_token


class AdaLNLayer(nn.Module):
    """
    Single transformer encoder layer with DiT-style Adaptive Layer Norm conditioning.
    Accepts batch_first tensors: x (B, L, E), conditioning c (B, E).

    At init, the AdaLN modulation is zero-initialised so every layer starts as an
    identity transformation (DiT training-stability trick).
    """
    def __init__(self, embed_dim, n_head, dim_feedforward, dropout, layer_norm_eps):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

        self.ff1 = nn.Linear(embed_dim, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, embed_dim)
        self.act = nn.GELU()
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff   = nn.Dropout(dropout)

        # No affine params — AdaLN provides scale and shift
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, elementwise_affine=False)

        # Maps c -> (shift1, scale1, gate1, shift2, scale2, gate2), each (B, E)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )
        # Zero-init: all modulations start at 0 → identity at initialisation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, src_key_padding_mask=None):
        # c: (B, E) → (B, 1, 6*E) for broadcasting over sequence length
        mod = self.adaLN_modulation(c).unsqueeze(1)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Attention sublayer with AdaLN pre-norm
        h = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask)
        x = x + gate1 * self.drop_attn(attn_out)

        # FFN sublayer with AdaLN pre-norm
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ff2(self.drop_ff(self.act(self.ff1(h))))
        x = x + gate2 * h

        return x

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

class TimestepTransformerClassifier(torch.nn.Module):
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
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sigma_map = TimestepEmbedder(embed_dim=embed_dim)

        # AdaLN-conditioned transformer layers (one per n_layers)
        self.layers = nn.ModuleList([
            AdaLNLayer(embed_dim, n_head, dim_feedforward, dropout, layer_norm_eps)
            for _ in range(n_layers)
        ])
        # Final output norm (standard in DiT)
        self.norm_out = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        PE = torch.zeros((max_len, embed_dim))
        pos = torch.arange(max_len).unsqueeze(-1)
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2) / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("PE", PE)
        
        self.Dropout = nn.Dropout(dropout)
        
        # Predictor head: a simple linear layer
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X: torch.Tensor, timestep: torch.Tensor):
        B, L = X.shape

        padding_mask = (X == PAD_token).to(X.device)  # (B, L)

        X = self.embedding(X)

        # Sinusoidal positional encoding
        X += self.PE[:L, :].unsqueeze(0)

        X = self.Dropout(X)

        # Timestep embedding: (B, E) — passed to each AdaLN layer, not added to X
        t = timestep.view(B)
        c = self.sigma_map(-torch.log(1 - (1 - self.sampling_eps) * t))

        # AdaLN-conditioned transformer layers
        for layer in self.layers:
            X = layer(X, c, src_key_padding_mask=padding_mask)
        X = self.norm_out(X)

        X = self.fc(X)
        return X
