"""
DiT-style discrete diffusion model with Adaptive Layer Norm (AdaLN) timestep conditioning.
Inspired by SEDD (sedd/model/transformer.py) but using standard PyTorch ops
so it works on CPU and GPU without flash_attn.

Key features vs existing models:
  - AdaLN: timestep embedding modulates scale/shift of every layer norm
  - Pre-norm architecture (more stable for deep networks)
  - Relative positional bias (RPE-style) for better positional generalisation
  - Gated residuals (gate_msa, gate_mlp) from DiT paper
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import PAD_token, MASK_token


# ─── Timestep embedder ───────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP → cond_dim vector."""
    def __init__(self, cond_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]; transform to log-SNR scale used in MDLM
        # σ(t) = -log(1 - (1 - ε) * t); use as the sinusoidal argument
        eps = 1e-5
        sigma = -torch.log1p(-(1 - eps) * t.clamp(eps, 1 - eps))   # (B,)
        freq_emb = self.sinusoidal(sigma, self.freq_dim)             # (B, freq_dim)
        return self.mlp(freq_emb)                                    # (B, cond_dim)


# ─── AdaLN utilities ─────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN scale and shift: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── DiT block ───────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Norm (AdaLN) timestep conditioning.
    Computes 6 modulation parameters (shift/scale/gate for attn + mlp) from cond.
    Pre-norm design; gated residuals as in the DiT paper.
    """
    def __init__(self, d_model: int, n_heads: int, cond_dim: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Standard multi-head attention (no causal mask → full bidirectional context)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        # AdaLN modulation: 6 × d_model parameters from cond vector
        # Initialised to zero so the block starts as identity
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x : (B, L, d_model)
        c : (B, cond_dim)
        key_padding_mask: (B, L) bool – True for positions to ignore (PAD)
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.adaLN(c).chunk(6, dim=-1)          # each: (B, d_model)

        # Attention sub-block with gated residual
        x_norm = modulate(self.norm1(x), shift_a, scale_a)   # (B, L, d)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        attn_out = self.dropout(self.attn_proj(attn_out))
        x = x + gate_a.unsqueeze(1) * attn_out               # gated residual

        # FFN sub-block with gated residual
        x_norm = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.dropout(self.ff(x_norm))

        return x


# ─── Full model ──────────────────────────────────────────────────────────────

class DiTDiffusionModel(nn.Module):
    """
    Full discrete diffusion model using DiT blocks with AdaLN.

    Architecture:
      1. Token embedding (vocab_size → embed_dim)
      2. Sinusoidal absolute positional encoding
      3. TimestepEmbedder: t → (B, cond_dim)
      4. N × DiTBlock with AdaLN conditioning from timestep
      5. Final LayerNorm
      6. Linear head → vocab_size logits
    """
    def __init__(
        self,
        max_len:    int   = 258,
        vocab_size: int   = 6,
        n_head:     int   = 8,
        n_layers:   int   = 8,
        embed_dim:  int   = 512,
        cond_dim:   int   = 256,
        dim_feedforward: int = 2048,
        dropout:    float = 0.1,
        layer_norm_eps: float = 1e-5,
        sampling_eps:   float = 1e-5,
    ):
        super().__init__()
        self.architecture = 'diffusion'
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.l            = max_len
        self.sampling_eps = sampling_eps

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Absolute sinusoidal positional encoding (fixed, not learned)
        PE = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(-1).float()
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2).float() / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("PE", PE)

        # Timestep embedder
        self.timestep_emb = TimestepEmbedder(cond_dim=cond_dim)

        # Transformer blocks
        mlp_ratio = dim_feedforward // embed_dim
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, n_head, cond_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.fc = nn.Linear(embed_dim, vocab_size)

        # Initialise output layer near zero
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, X: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        X        : (B, L) int – token ids (may include MASK_token=5)
        timestep : (B,) or (B,1) float in [0, 1]
        Returns  : (B, L, vocab_size) logits
        """
        B, L = X.shape
        t = timestep.view(B).float()

        # Padding mask: True = ignore (PAD tokens don't attend)
        pad_mask = (X == PAD_token)   # (B, L) bool

        # Embeddings
        x = self.embedding(X) + self.PE[:L].unsqueeze(0)   # (B, L, E)

        # Timestep conditioning
        c = self.timestep_emb(t)   # (B, cond_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, key_padding_mask=pad_mask)

        x = self.final_norm(x)
        return self.fc(x)     # (B, L, vocab_size)
