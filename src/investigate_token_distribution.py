import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from constants import MASK_token, PAD_token, SOS_token, EOS_token
from deterministic_token_distribution import determineTokenDistribution
from model_RPE import RPETransformerClassifier
from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation


def investigate_seq(model, unmasker, device, grammar, seq, figures_dir,
                    failing_ID, correct_ID, n_first_tokens=10**9, store_numeric=False):
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

        numeric_log = group_dir / f'{case_label}_numeric.txt'
        plot_path = group_dir / f'{case_label}_kl_plot.png'
        
        if store_numeric:
            # ---- numeric log header ----
            with open(numeric_log, 'w') as f:
                f.write(f'START for sequence: {input_seq.tolist()}\n')
                f.write(f'Final sequence:     {final_1d.tolist()}\n')
                f.write(f'Evaluation (r1, r2, both, fmt): {grammar.evaluate(final_1d).tolist()}\n\n')

        # ---- per-step loop ----
        prev_timestep = -1.0
        kl_divergences  = []
        timestep_values = []

        for idx, seq_step in enumerate(steps):
            if idx == len(steps) - 1:  # skip fully-unmasked final state
                continue

            timestep = torch.tensor(timesteps[idx])
            if (seq_step == MASK_token).sum() == (steps[idx-1] == MASK_token).sum() if idx > 0 else False:
                continue  # timestep unchanged — no new tokens were revealed
            prev_timestep = timestep.item()

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
                with open(numeric_log, 'a') as f:
                    f.write(f'No valid completion exists after {idx} step(s). Stopping.\n')
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
                with open(numeric_log, 'a') as f:
                    f.write(f'Timestep: {timestep.item():.4f},  KL Divergence: {div:.4f}\n')
                    f.write(f'Sequence before:  {dt_seq.tolist()}\n')
                    f.write(f'Sequence after:  {steps[idx + 1].tolist()}\n')

                    # Iterate only over masked positions where either
                    # predicted confidence < 0.6 OR argmax(pred) != argmax(expected).
                    masked_positions = (dt_seq == MASK_token)
                    low_conf = pred_conf_per_pos < 0.6
                    argmax_mismatch = pred_argmax_per_pos != exp_argmax_per_pos
                    positions_to_log = torch.where(masked_positions & (low_conf | argmax_mismatch))[0]

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
                            f'Position {pos} (token={token_val})  KL={kl:.4f}  '
                            f'Entropy(exp)={H_exp:.4f}  Entropy(pred)={H_pred:.4f}  Confidence={pred_max:.4f}\n'
                        )
                        f.write(f'  Expected:  [{exp_str}]\n')
                        f.write(f'  Predicted: [{pred_str}]\n')
                    f.write('\n')

        if store_numeric:
            with open(numeric_log, 'a') as f:
                f.write(f'FINISH for sequence: {input_seq.tolist()}\n')

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
            n_first_tokens=n_first_tokens, store_numeric=store_numeric
        )