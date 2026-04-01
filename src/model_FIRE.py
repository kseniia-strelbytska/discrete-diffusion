
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class FireSelfAttention(nn.Module):
    """
    From: https://github.com/meta-pytorch/torchtune/pull/2388
    
    This class implements FIRE (Functional Interpolation for Relative Positional Encodings)
    as described in https://arxiv.org/abs/2310.04418 for causal language modeling tasks. The
    only modification from the paper is that this implementation uses the GELU activation function instead
    of ReLU in order to avoid possible problems with "dying" neurons.

    Args:
        dim_model (int): The embedding dimension of the input vectors.
        num_heads (int): The number of self-attention heads, set to 1 by default. The dimension of each individual head
            is usually computed as ``dim_model // num_heads``.
        hidden_size (int): The dimension of the MLP layers in each attention head used to compute the bias matrix.

    Note: This module is fundamentally a positional encoding scheme; however, due to the nature of FIRE relative
        positional encodings, it takes the form of an attention layer.
    """

    def __init__(
        self, dim_model: int, num_heads: int = 1, hidden_size: int = 32
    ) -> None:
        super().__init__()

        # make sure num_heads divides dim_model:
        assert (
            dim_model % num_heads == 0
        ), "Number of heads must divide dimension of model"

        # compute kdim = vdim
        kdim = dim_model // num_heads

        # initialize attention heads
        self.attention_heads = nn.ModuleList(
            [
                self.FireAttentionHead(dim_model, kdim, hidden_size)
                for _ in range(num_heads)
            ]
        )

        # final linear layer
        self.W_o = nn.Linear(dim_model, dim_model, bias=False)

    class FireAttentionHead(nn.Module):
        """
        An inner class to implement a single attention head using the FIRE positional encoding scheme.
        **Do not** use this class directly; instead use FireSelfAttention with ``num_heads = 1`` if you need it.

        Args:
            dim_model (int): The embedding dimension of the input vectors, as above.
            kdim (int): The dimension of the query, key, and value vectors, computed as ``kdim = dim_model // num_heads``.
            hidden_size (int): The dimension of the MLP layers in each attention head used to compute the bias matrix.
        """

        def __init__(self, dim_model: int, kdim: int, hidden_size: int) -> None:
            super().__init__()
            self.kdim = kdim

            # initialize parameter matrices
            self.W_q = nn.Linear(dim_model, kdim, bias=False)
            self.W_k = nn.Linear(dim_model, kdim, bias=False)
            self.W_v = nn.Linear(dim_model, kdim, bias=False)

            # initialize learnable scalars to "reasonable" values (these are arbitary and can be adjusted later on.)
            # c is used to modify the input of the logarithm in the phi function.
            self.c = nn.Parameter(torch.tensor(1.0))
            # L is used in the adaptive thresholding mechanism to activate progressive interpolation only for long contexts.
            self.L = nn.Parameter(torch.tensor(2.0))

            # initialize learnable continuous function
            self.f_theta = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        # concave function to amplify differences among local positions
        def phi(self, c: nn.Parameter, x: torch.Tensor) -> torch.Tensor:
            return torch.log1p(c * x)

        def forward(self, src: torch.Tensor) -> torch.Tensor:
            """
            Args:
                src (torch.Tensor): Input tensor with shape ``[batch_size, seq_length, dim_model]``

            Returns:
                torch.Tensor: Output tensor of shape ``[batch_size, seq_length, kdim]``
            """
            # Assuming src has shape (batch_size, seq_length, dim_model)
            batch_size, seq_length = src.shape[0:2]

            # constrain c to be > 0
            c = torch.nn.functional.softplus(self.c)

            # compute bias matrix using vectorized operations
            # below, i is the query position and j is the key position, 0 <= i - j < i
            positions = torch.arange(seq_length, device=src.device).unsqueeze(1)
            positions_diff = positions - torch.arange(seq_length, device=src.device).unsqueeze(0)
            # Create lower triangular mask for causal attention
            causal_mask = positions_diff >= 0
            positions_diff = positions_diff.float()
            
            # Compute numerator: phi(c, i - j)
            numerator = self.phi(c, positions_diff)
            # Compute denominator: phi(c, max(L, i + 1))
            denom_positions = torch.maximum(self.L, positions + 1)
            denominator = self.phi(c, denom_positions)
            
            # Compute bias
            bias = numerator / denominator
            bias = bias * causal_mask.float()  # Apply causal mask
            # apply MLP to bias matrix
            bias = self.f_theta(bias.unsqueeze(2)).squeeze(2)
            # add causal mask
            lookahead_mask = torch.ones(seq_length, seq_length, dtype=torch.bool).triu(
                diagonal=1
            ).to(src.device)
            bias.masked_fill_(lookahead_mask, float("-inf"))
            # repeat bias matrix for batch_size
            bias = bias.repeat(batch_size, 1, 1)

            # get Query, Key, and Value matrices for each sequence
            q = self.W_q(src)
            k = self.W_k(src)
            v = self.W_v(src)

            # calculate attention scores
            k_t = torch.transpose(k, 1, 2)
            attn_logits = torch.bmm(q, k_t) / (self.kdim**0.5)
            attn_logits = attn_logits + bias
            attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
            attn_outputs = torch.bmm(attn_weights, v)
            return attn_outputs

    # End of the inner class for a single attention head

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input tensor with shape ``[batch_size, seq_length, dim_model]``

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_length, dim_model]`` with multi-head attention
            and FIRE relative positional encoding applied.
        """
        # src should have shape (batch_size, seq_length, dim_model)
        # Pass src through the attention heads
        attn_results = [attn_head(src) for attn_head in self.attention_heads]
        # concatenate results
        attn_results = torch.cat(attn_results, dim=-1)
        # pass through final linear layer
        return self.W_o(attn_results)
    
class TransformerBlock(nn.Module):
    """A standard Transformer block using FIRE Self Attention."""
    def __init__(self, embed_dim, n_head, dim_feedforward, fire_hidden_size=32):
        super().__init__()
        # 1. Attention with a SMALL hidden size for the positional MLP
        self.attention = FireSelfAttention(
            dim_model=embed_dim, 
            num_heads=n_head, 
            hidden_size=fire_hidden_size 
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 2. The actual Feed-Forward Network using dim_feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Standard residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FIRETransformerClassifier(torch.nn.Module):
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

        # Build network using the new Transformer blocks
        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(
                embed_dim=embed_dim, 
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                fire_hidden_size=32 # Keep this small!
            ) for _ in range(n_layers)]
        )

        # Predictor head: a simple linear layer
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X: torch.Tensor, timestep: torch.Tensor = None):
        # B, L = X.shape
        X = self.embedding(X)  
        
        # Pass through network
        X = self.transformer_encoder(X)
        X = self.fc(X)

        return X