import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import base64
import io
import html
from pathlib import Path
from tqdm import tqdm

from constants import MASK_token, PAD_token, SOS_token, EOS_token
from deterministic_token_distribution import determineTokenDistribution
from model_RPE import RPETransformerClassifier
from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from attention_maps import attach_attention_hooks, extract_attention_maps, remove_hooks


def _attention_grid_to_base64(attn_maps, seq_tokens, max_tokens=24):
    """Render all layer/head attention maps into one figure and return base64 PNG."""
    if seq_tokens.dim() == 2:
        seq_tokens = seq_tokens.squeeze(0)
    token_labels = [str(int(t)) for t in seq_tokens.tolist()]
    max_tokens = min(max_tokens, len(token_labels))
    token_labels = token_labels[:max_tokens]

    n_layers = len(attn_maps)
    if n_layers == 0:
        return None
    n_heads = max(attn.shape[1] for attn in attn_maps)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.3 * n_heads, 3.1 * n_layers), squeeze=False)

    for layer_idx, attn in enumerate(attn_maps):
        layer_attn = attn[0].detach().cpu()  # (nhead, L, L)
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]

            if head_idx >= layer_attn.shape[0]:
                ax.axis('off')
                continue

            matrix = layer_attn[head_idx].numpy()[:max_tokens, :max_tokens]
            vmax = float(matrix.max()) if matrix.size > 0 else 1.0
            if vmax <= 0:
                vmax = 1.0
            ax.imshow(matrix, cmap='viridis', vmin=0.0, vmax=vmax, aspect='auto')

            if max_tokens <= 20:
                ax.set_xticks(range(len(token_labels)))
                ax.set_yticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, fontsize=6, rotation=90)
                ax.set_yticklabels(token_labels, fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f'Layer {layer_idx} / Head {head_idx}', fontsize=8)
            ax.set_xlabel('Key token ID', fontsize=7)
            ax.set_ylabel('Query token ID', fontsize=7)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def investigate_seq(model, unmasker, device, grammar, seq, figures_dir,
                    failing_ID, correct_ID, n_first_tokens=10**9, store_numeric=False,
                    attention_every=10, max_attention_tokens=24):
    """
    Investigate a single sequence.
    
    Returns
    -------
    (failing_ID, correct_ID) : both counters, updated if this seq was failing/correct.
    """
    PRECISION = 4

    with torch.no_grad():
        model.eval()
        # Unmasker accepts 1D sequences.
        input_seq = torch.tensor(seq).long() if not isinstance(seq, torch.Tensor) else seq.long()
        timestep_init = ((input_seq == MASK_token).sum() / torch.numel(input_seq))
        final, steps, timesteps = unmasker(input_seq, timestep_init, return_steps=True)

        # Evaluate the final denoised sequence to determine pass / fail.
        final_1d = final if final.dim() == 1 else final.squeeze(0)
        is_failing = (grammar.evaluate(final_1d) == 0).sum() > 0

        # create flat directories under figures_dir: figures_dir/Correct and figures_dir/Failing
        group_name = 'Failing' if is_failing else 'Correct'
        group_dir = figures_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        if is_failing:
            failing_ID += 1
            case_label = f'FAILING_CASE_{failing_ID}'
        else:
            correct_ID += 1
            case_label = f'CORRECT_CASE_{correct_ID}'

        numeric_log = group_dir / f'{case_label}_numeric.html'
        plot_path = group_dir / f'{case_label}_kl_plot.png'

        attention_hooks = None
        attention_msg = None

        if store_numeric:
            try:
                attention_hooks = attach_attention_hooks(model)
            except ValueError as e:
                attention_msg = str(e)
            with open(numeric_log, 'w') as f:
                f.write("""
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Investigation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 16px; }
        .meta { background:#f6f8fa; padding:10px; border-radius:8px; margin-bottom:12px; }
        .step { border:1px solid #ddd; border-radius:8px; padding:12px; margin:12px 0; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
        img { max-width: 100%; height: auto; border:1px solid #ccc; border-radius:6px; }
    </style>
</head>
<body>
""")
                f.write(f"<h2>{html.escape(case_label)} investigation</h2>")
                f.write("<div class='meta mono'>")
                f.write(f"START for sequence: {html.escape(str(input_seq.tolist()))}<br/>")
                f.write(f"Final sequence: {html.escape(str(final_1d.tolist()))}<br/>")
                f.write(f"Evaluation (r1, r2, both, fmt): {html.escape(str(grammar.evaluate(final_1d).tolist()))}<br/>")
                if attention_msg is not None:
                    f.write(f"Attention maps unavailable for this architecture: {html.escape(attention_msg)}<br/>")
                else:
                    f.write(
                        f"Attention snapshots sampled every {attention_every} denoising step(s); "
                        f"rendering up to first {max_attention_tokens} tokens per axis.<br/>"
                    )
                f.write("</div>")

        # ---- per-step loop ----
        kl_divergences  = []
        timestep_values = []

        for idx, seq_step in enumerate(steps):
            if idx == len(steps) - 1:  # skip fully-unmasked final state
                continue

            timestep = timesteps[idx].detach().clone() if isinstance(timesteps[idx], torch.Tensor) else torch.tensor(timesteps[idx], device=device)
            if (seq_step == MASK_token).sum() == (steps[idx-1] == MASK_token).sum() if idx > 0 else False:
                continue  # timestep unchanged — no new tokens were revealed

            # Model expects (B, L); ensure batch dim.
            model_input    = seq_step.unsqueeze(0) if seq_step.dim() == 1 else seq_step
            timestep_input = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep

            predicted_logits = model(model_input, timestep_input).squeeze(0)
            predicted_log_probs = F.log_softmax(predicted_logits, dim=-1)
            predicted_distribution = predicted_log_probs.exp()

            # determineTokenDistribution expects a 1D sequence.
            dt_seq = seq_step if seq_step.dim() == 1 else seq_step.squeeze(0)
            expected_distribution = determineTokenDistribution(dt_seq, vocab_size=model.vocab_size)

            if expected_distribution[0] is None:
                if store_numeric:
                    with open(numeric_log, 'a') as f:
                        f.write(f"<div class='step mono'>No valid completion exists after {idx} step(s). Stopping.</div>")
                break

            expected_dist_tensor = expected_distribution[1][:n_first_tokens].to(device)

            # Vectorized per-position metrics over full (L, vocab) tensors.
            eps = 1e-8
            pred_probs = predicted_distribution
            exp_probs = expected_dist_tensor

            kl_per_pos = (exp_probs * (torch.log(exp_probs + eps) - torch.log(pred_probs + eps))).sum(dim=-1)
            H_exp_per_pos = -(exp_probs * torch.log(exp_probs + eps)).sum(dim=-1)
            H_pred_per_pos = -(pred_probs * torch.log(pred_probs + eps)).sum(dim=-1)
            pred_conf_per_pos = pred_probs.max(dim=-1).values
            pred_argmax_per_pos = pred_probs.argmax(dim=-1)
            exp_argmax_per_pos = exp_probs.argmax(dim=-1)

            # Keep global KL trend for the per-step line plot.
            div = kl_per_pos.mean().item()

            kl_divergences.append(div)
            timestep_values.append(timestep.item())

            if store_numeric:
                # Iterate only over masked positions where either
                # predicted confidence < 0.6 OR argmax(pred) != argmax(expected).
                masked_positions = (dt_seq == MASK_token)
                low_conf = pred_conf_per_pos < 0.6
                argmax_mismatch = pred_argmax_per_pos != exp_argmax_per_pos
                positions_to_log = torch.where(masked_positions & (low_conf | argmax_mismatch))[0]
                with open(numeric_log, 'a') as f:
                    f.write("<div class='step'>")
                    f.write("<div class='mono'>")
                    f.write(f"Timestep: {timestep.item():.4f}, KL Divergence: {div:.4f}<br/>")
                    f.write(f"Sequence before: {html.escape(str(dt_seq.tolist()))}<br/>")
                    f.write(f"Sequence after: {html.escape(str(steps[idx + 1].tolist()))}<br/><br/>")

                    for pos in positions_to_log.tolist():
                        token_val = dt_seq[pos].item()
                        kl = kl_per_pos[pos].item()
                        H_exp = H_exp_per_pos[pos].item()
                        H_pred = H_pred_per_pos[pos].item()
                        pred_max = pred_conf_per_pos[pos].item()

                        exp_row = exp_probs[pos][:6].tolist()
                        pred_row = pred_probs[pos][:6].tolist()

                        exp_str = ', '.join([f'{x:.4f}' for x in exp_row])
                        pred_str = ', '.join([f'{x:.4f}' for x in pred_row])

                        f.write(
                            f"Position {pos} (token={token_val})  KL={kl:.4f}  "
                            f"Entropy(exp)={H_exp:.4f}  Entropy(pred)={H_pred:.4f}  Confidence={pred_max:.4f}<br/>"
                        )
                        f.write(f"&nbsp;&nbsp;Expected:&nbsp;&nbsp;[{exp_str}]<br/>")
                        f.write(f"&nbsp;&nbsp;Predicted: [{pred_str}]<br/>")

                    f.write("</div>")

                    # Lightweight mode: render attention only on sampled steps and only when there are flagged positions.
                    should_render_attention = (
                        attention_hooks is not None
                        and positions_to_log.numel() > 0
                        and (idx % max(1, attention_every) == 0)
                    )
                    if should_render_attention:
                        attn_maps = extract_attention_maps(
                            model,
                            dt_seq,
                            device=device,
                            timestep=timestep_input,
                        )
                        b64_img = _attention_grid_to_base64(
                            attn_maps,
                            dt_seq,
                            max_tokens=max_attention_tokens,
                        )
                        if b64_img is not None:
                            f.write("<h4>Attention maps (all layers/heads)</h4>")
                            f.write(f"<img alt='attention maps' src='data:image/png;base64,{b64_img}' />")

                    f.write("</div>")

        if store_numeric:
            with open(numeric_log, 'a') as f:
                f.write(f"<div class='meta mono'>FINISH for sequence: {html.escape(str(input_seq.tolist()))}</div>")
                f.write("</body></html>")

            if attention_hooks is not None:
                remove_hooks(attention_hooks)

        # ---- KL divergence line plot ----
        if kl_divergences:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(timestep_values, kl_divergences, marker='o', linewidth=1.5, markersize=4)
            ax.set_xlabel('Timestep  (right = more denoised)')
            ax.set_ylabel('KL Divergence')
            ax.set_title(f'KL Divergence over Denoising Steps\n{case_label}')
            ax.invert_xaxis()  # denoising goes high → low timestep; show progress left→right
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

    return failing_ID, correct_ID


def investigate_dataset(model, unmasker, device, grammar, dataset, figures_dir, n_first_tokens=10**9, store_numeric=False):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    failing_ID = 0
    correct_ID = 0

    for i in tqdm(range(len(dataset))):
        seq = dataset[i]
        failing_ID, correct_ID = investigate_seq(
            model, unmasker, device=device, grammar=grammar, seq=seq,
            figures_dir=figures_dir, failing_ID=failing_ID, correct_ID=correct_ID,
            n_first_tokens=n_first_tokens, store_numeric=store_numeric,
            attention_every=10, max_attention_tokens=24
        )