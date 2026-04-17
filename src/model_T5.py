"""

TransformerClassifier with T5-style Relative Position Bias
instead of Music-Transformer-style skew RPE.

Key change: relative positions map to a small learned scalar bias
per head, added directly to attention logits before softmax.
Q, K, V are untouched by positional encoding.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class T5MultiheadAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model=128,
        max_len=16,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        layer_norm_eps=2e-4,
        batch_first=True,
        norm_first=False,
        num_buckets=600,         # number of distance buckets (T5 default: 32)
        bidirectional=True,     # True for encoder, False for causal decoder
    ):
        super().__init__()

        self.max_len = max_len
        self.nhead = nhead
        self.norm_first = norm_first
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional

        self.qkv = nn.Linear(d_model, 3 * d_model)

        # T5 bias: one scalar per (head, bucket) — tiny table
        self.rel_pos_bias = nn.Embedding(num_buckets, nhead)

        self.proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)

        self.relu = nn.ReLU()
        self.norm_attn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_ff = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout_actv = nn.Dropout(dropout)
        self.dropout_ff1 = nn.Dropout(dropout)
        self.dropout_ff2 = nn.Dropout(dropout)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Map integer relative positions to bucket indices.

        Mirrors the T5 bucketing scheme:
          - First half of buckets: exact distances 0..num_buckets//2
          - Second half:           log-scale beyond that, up to max_len

        Args:
            relative_position: (L, L) integer tensor of (i - j) values

        Returns:
            bucket_ids: (L, L) long tensor in [0, num_buckets)
        """
        num_buckets = self.num_buckets
        max_distance = self.max_len

        # For bidirectional (encoder): treat positive and negative separately,
        # using half the buckets for each direction.
        # For causal (decoder): ignore future positions (they will be masked anyway).
        if self.bidirectional:
            num_buckets //= 2
            # Offset positive relative positions into the upper half of the table
            bucket_ids = (relative_position > 0).long() * num_buckets
            relative_position = relative_position.abs()
        else:
            # Causal: clip negatives to 0 (past only)
            relative_position = -relative_position.clamp(max=0)
            bucket_ids = torch.zeros_like(relative_position)

        # Split into exact (small distances) vs log-scale (large distances)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-scale bucket for large distances
        val_if_large = max_exact + (
            torch.log(relative_position.float().clamp(min=1) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long().clamp(max=num_buckets - 1)

        bucket_ids += torch.where(is_small, relative_position, val_if_large)
        return bucket_ids  # (L, L)

    def _compute_bias(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Build the (1, nhead, L, L) additive bias for a sequence of length L.
        Computed once per forward pass; cheap for small L.
        """
        positions = torch.arange(L, device=device)
        # relative_position[i, j] = i - j
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)

        bucket_ids = self._relative_position_bucket(relative_position)       # (L, L)

        # bias shape: (L, L, nhead) → (1, nhead, L, L)
        bias = self.rel_pos_bias(bucket_ids)                                  # (L, L, nhead)
        bias = bias.permute(2, 0, 1).unsqueeze(0)                            # (1, nhead, L, L)
        return bias

    def _sa_block(self, X: torch.Tensor) -> torch.Tensor:
        B, L, _ = X.shape

        # Project to Q, K, V — no positional information injected here
        qkv = self.qkv(X)
        qkv = qkv.reshape(B, L, 3, self.nhead, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, nhead, L, d_head)
        Q, K, V = qkv.unbind(0)                     # each: (B, nhead, L, d_head)

        # Content-based attention logits
        attn = Q @ K.transpose(-2, -1)              # (B, nhead, L, L)
        attn = attn / math.sqrt(Q.shape[-1])

        # T5-style additive positional bias (Q, K, V untouched)
        bias = self._compute_bias(L, X.device)      # (1, nhead, L, L)
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)              # (B, nhead, L, L)

        out = attn @ V                              # (B, nhead, L, d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)  # (B, L, d_model)
        out = self.dropout_actv(self.proj(out))

        return out

    def _ff_block(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout_ff2(self.ff2(self.dropout_ff1(self.relu(self.ff1(X)))))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            X = X + self._sa_block(self.norm_attn(X))
            X = X + self._ff_block(self.norm_ff(X))
        else:
            X = self.norm_attn(X + self._sa_block(X))
            X = self.norm_ff(X + self._ff_block(X))
        return X


class T5RPETransformerClassifier(nn.Module):
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
        num_buckets=32,
        bidirectional=True,
    ):
        super().__init__()
        self.architecture = 'diffusion'

        self.l = max_len
        self.sampling_eps = sampling_eps
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.transformer_encoder = nn.Sequential(
            *[T5MultiheadAttentionLayer(
                d_model=embed_dim,
                max_len=max_len,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=False,
                num_buckets=num_buckets,
                bidirectional=bidirectional,
            ) for _ in range(n_layers)]
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)           # (B, L, embed_dim)
        X = self.transformer_encoder(X)
        X = self.fc(X)                  # (B, L, vocab_size)
        return X