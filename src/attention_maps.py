"""
Standalone attention-map extraction & visualisation.

Supports architectures with custom self-attention blocks:
  - RPETransformerClassifier  (model_RPE.py)
  - v2TransformerClassifier   (model_v2.py)
  - FIRETransformerClassifier (model_FIRE.py)

Models that use PyTorch's native nn.TransformerEncoder / nn.MultiheadAttention
(classic, timestep, autoregressive, RE) do NOT expose attention weights without
modifying the source, so they are **not** supported.

Usage (from main.py investigate branch):
    from attention_maps import (
        attach_attention_hooks,
        extract_attention_maps,
        plot_attention_maps,
    )
    hooks = attach_attention_hooks(model)          # patches every supported layer
    maps  = extract_attention_maps(model, seq, device)
    plot_attention_maps(maps, seq, save_dir)
    # When done, call remove_hooks(hooks) to restore original behaviour.
"""

import types
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# 1. Monkey-patching helpers
# ---------------------------------------------------------------------------

def _patched_sa_block_RPE(self, X):
    """Drop-in replacement for RPEMultiheadAttentionLayer._sa_block.
    Executes the *exact* original logic, but saves `attn` to self.saved_attn_map."""
    B, L, _ = X.shape

    qkv = self.qkv(X)
    qkv = qkv.reshape(X.shape[0], X.shape[1], 3, self.nhead, -1)
    qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, nhead, L, d_k)
    Q, K, V = qkv.unbind(0)                  # each (B, nhead, L, d_k)

    R = self.R_embed(
        torch.arange(start=self.max_len - L, end=self.max_len + L - 1).to(X.device)
    )  # (2L-1, d_k)

    skew = Q @ R[None, None, :, :].transpose(-2, -1)

    skew = F.pad(skew, (0, 1))                                   # (B, nhead, L, 2L)
    skew = skew.view(B, self.nhead, L * 2 * L)                   # (B, nhead, 2L^2)
    skew = F.pad(skew, (0, L - 1))                               # (B, nhead, 2L^2+L-1)
    skew = skew.view(B, self.nhead, L + 1, 2 * L - 1)            # (B, nhead, L+1, 2L-1)
    skew = skew[:, :, :L, L - 1:]                                # (B, nhead, L, L)

    attn = (Q @ K.transpose(-2, -1) + skew) / math.sqrt(Q.shape[-1])
    attn = F.softmax(attn, dim=-1)            # (B, nhead, L, L)

    # >>> the only addition <<<
    self.saved_attn_map = attn.detach()

    out = attn @ V
    out = out.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[1], -1)
    out = self.dropout_actv(self.proj(out))
    return out


def _patched_sa_block_v2(self, X):
    """Drop-in replacement for v2MultiheadAttentionLayer._sa_block."""
    qkv = self.qkv(X)
    qkv = qkv.reshape(X.shape[0], X.shape[1], 3, self.nhead, -1)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    Q, K, V = qkv.unbind(0)

    attn = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    attn = F.softmax(attn, dim=-1)

    self.saved_attn_map = attn.detach()

    out = attn @ V
    out = out.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[1], -1)
    out = self.dropout_actv(self.proj(out))
    return out


def _patched_fire_head_forward(self, src):
    """Drop-in replacement for FireAttentionHead.forward."""
    batch_size, seq_length = src.shape[0:2]

    c = F.softplus(self.c)

    positions = torch.arange(seq_length, device=src.device).unsqueeze(1)
    positions_diff = positions - torch.arange(seq_length, device=src.device).unsqueeze(0)
    causal_mask = positions_diff >= 0
    positions_diff = positions_diff.float()

    numerator = self.phi(c, positions_diff)
    denom_positions = torch.maximum(self.L, positions + 1)
    denominator = self.phi(c, denom_positions)

    bias = numerator / denominator
    bias = bias * causal_mask.float()
    bias = self.f_theta(bias.unsqueeze(2)).squeeze(2)

    lookahead_mask = torch.ones(seq_length, seq_length, dtype=torch.bool).triu(diagonal=1).to(src.device)
    bias.masked_fill_(lookahead_mask, float("-inf"))
    bias = bias.repeat(batch_size, 1, 1)

    q = self.W_q(src)
    k = self.W_k(src)
    v = self.W_v(src)

    k_t = torch.transpose(k, 1, 2)
    attn_logits = torch.bmm(q, k_t) / (self.kdim ** 0.5)
    attn_logits = attn_logits + bias
    attn_weights = F.softmax(attn_logits, dim=-1)

    # >>> the only addition <<<
    self.saved_attn_map = attn_weights.detach()

    attn_outputs = torch.bmm(attn_weights, v)
    return attn_outputs


# ---------------------------------------------------------------------------
# 2. Hook attachment / removal
# ---------------------------------------------------------------------------

def _get_architecture_tag(model):
    """Return a tag string based on the model class name."""
    cls = type(model).__name__
    if cls == "RPETransformerClassifier":
        return "RPE"
    if cls == "v2TransformerClassifier":
        return "v2"
    if cls == "FIRETransformerClassifier":
        return "FIRE"
    return "unsupported"


def attach_attention_hooks(model):
    """Monkey-patch every supported attention layer in `model`.

    Returns a list of (layer, original_method) tuples so we can restore later.
    """
    tag = _get_architecture_tag(model)
    originals = []

    if tag == "RPE":
        for layer in model.transformer_encoder:
            originals.append((layer, layer._sa_block))
            layer._sa_block = types.MethodType(_patched_sa_block_RPE, layer)

    elif tag == "v2":
        for layer in model.transformer_encoder:
            originals.append((layer, layer._sa_block))
            layer._sa_block = types.MethodType(_patched_sa_block_v2, layer)

    elif tag == "FIRE":
        for block in model.transformer_encoder:
            for head in block.attention.attention_heads:
                originals.append((head, head.forward))
                head.forward = types.MethodType(_patched_fire_head_forward, head)

    else:
        raise ValueError(
            f"Attention-map extraction is not supported for {type(model).__name__}. "
            "Only RPE, v2, and FIRE architectures with custom attention blocks "
            "are supported. Models using PyTorch's native nn.MultiheadAttention "
            "(classic, timestep, autoregressive, RE) do not expose attention weights."
        )

    return originals


def remove_hooks(originals):
    """Restore the original methods saved by `attach_attention_hooks`."""
    for obj, original_method in originals:
        # Determine which attribute was patched
        if hasattr(original_method, "__name__") and original_method.__name__ == "forward":
            obj.forward = original_method
        else:
            obj._sa_block = original_method


# ---------------------------------------------------------------------------
# 3. Extraction
# ---------------------------------------------------------------------------

def extract_attention_maps(model, seq, device, timestep=None):
    """Run a forward pass and return the saved attention maps.

    Parameters
    ----------
    model : nn.Module
        A model whose hooks have already been attached.
    seq : torch.Tensor
        Token-id sequence of shape (L,) or (1, L).
    device : torch.device
    timestep : float | torch.Tensor | None
        If None a zero timestep is used.

    Returns
    -------
    list[torch.Tensor]
        One tensor per layer.  Shape depends on architecture:
        - RPE / v2:  (1, nhead, L, L)
        - FIRE: list of per-head tensors (1, L, L) grouped by block.
    """
    model.eval()
    tag = _get_architecture_tag(model)

    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    seq = seq.to(device)

    if timestep is None:
        timestep = torch.zeros(1, device=device)
    elif not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=device)

    with torch.no_grad():
        _ = model(seq, timestep)

    maps = []
    if tag in ("RPE", "v2"):
        for layer in model.transformer_encoder:
            maps.append(layer.saved_attn_map)  # (1, nhead, L, L)
    elif tag == "FIRE":
        for block in model.transformer_encoder:
            # stack per-head maps → (1, nhead, L, L) for consistency
            head_maps = [
                head.saved_attn_map for head in block.attention.attention_heads
            ]
            stacked = torch.stack(head_maps, dim=1)  # (B, nhead, L, L)
            maps.append(stacked)

    return maps


# ---------------------------------------------------------------------------
# 4. Visualisation
# ---------------------------------------------------------------------------

def plot_attention_maps(attn_maps, seq, save_dir, title_prefix="",
                        row_range=None, col_range=None):
    """Plot and save attention heatmaps, optionally zoomed into a sub-region.

    Parameters
    ----------
    attn_maps : list[torch.Tensor]
        Output of `extract_attention_maps`.  Each tensor has shape (1, nhead, L, L).
    seq : torch.Tensor
        The 1-D token-id sequence used during the forward pass.
    save_dir : str | Path
        Directory to save the figures into.
    title_prefix : str
        Optional prefix for plot titles / filenames.
    row_range : tuple(int, int) | None
        If given, slice rows (query positions) as ``matrix[row_range[0]:row_range[1]]``.
    col_range : tuple(int, int) | None
        If given, slice columns (key positions) as ``matrix[:, col_range[0]:col_range[1]]``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if seq.dim() == 2:
        seq = seq.squeeze(0)
    all_labels = [str(tok.item()) for tok in seq]

    # Determine label subsets for potentially zoomed axes.
    row_labels = all_labels[row_range[0]:row_range[1]] if row_range is not None else all_labels
    col_labels = all_labels[col_range[0]:col_range[1]] if col_range is not None else all_labels

    zoomed = row_range is not None or col_range is not None

    for layer_idx, attn in enumerate(attn_maps):
        attn = attn[0]  # remove batch dim → (nhead, L, L)
        nhead = attn.shape[0]

        fig, axes = plt.subplots(1, nhead, figsize=(4 * nhead, 4))
        if nhead == 1:
            axes = [axes]

        for head_idx, ax in enumerate(axes):
            matrix = attn[head_idx].cpu().numpy()  # (L, L)

            # Apply optional zoom slicing.
            r0, r1 = (row_range[0], row_range[1]) if row_range is not None else (0, matrix.shape[0])
            c0, c1 = (col_range[0], col_range[1]) if col_range is not None else (0, matrix.shape[1])
            matrix = matrix[r0:r1, c0:c1]

            zoomed_rows, zoomed_cols = matrix.shape
            annotate = zoomed and (zoomed_rows < 20 and zoomed_cols < 20)

            if HAS_SEABORN:
                sns.heatmap(
                    matrix,
                    ax=ax,
                    xticklabels=col_labels,
                    yticklabels=row_labels,
                    cmap="viridis",
                    vmin=0,
                    vmax=matrix.max(),
                    square=True,
                    cbar=True,
                    cbar_kws={"shrink": 0.6},
                    annot=annotate,
                    fmt=".3f" if annotate else "",
                    annot_kws={"size": 7} if annotate else {},
                )
            else:
                im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, fontsize=7)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=7)
                fig.colorbar(im, ax=ax, shrink=0.6)
                if annotate:
                    for r in range(zoomed_rows):
                        for c in range(zoomed_cols):
                            ax.text(c, r, f"{matrix[r, c]:.3f}",
                                    ha="center", va="center", fontsize=6, color="white")

            ax.set_xlabel("Key position (token ID)")
            ax.set_ylabel("Query position (token ID)")
            ax.set_title(f"Head {head_idx}")

        zoom_desc = ""
        if row_range is not None or col_range is not None:
            r_str = f"rows{r0}-{r1}" if row_range is not None else ""
            c_str = f"cols{c0}-{c1}" if col_range is not None else ""
            zoom_desc = "_zoom_" + "_".join(filter(None, [r_str, c_str]))

        fig.suptitle(
            f"{title_prefix}Layer {layer_idx} attention"
            + (f"  [rows {r0}:{r1}, cols {c0}:{c1}]" if zoom_desc else ""),
            fontsize=13,
            y=1.02,
        )
        fig.tight_layout()

        base = f"{title_prefix}layer_{layer_idx}_attention" if title_prefix else f"layer_{layer_idx}_attention"
        fname = f"{base}{zoom_desc}.png"
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
