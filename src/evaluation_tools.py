import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import html
from tqdm import tqdm
from noise_schedule_unmask import ScheduledUnmasker
from constants import EOS_token, SOS_token, PAD_token, MASK_token
from anbn import anbnGrammar
from dataset import Dataset, get_fixed_dataset
from AR_generation_and_predictions import get_prediction
from deterministic_token_distribution import oracleModel, determineTokenDistribution
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
        layer_attn = attn[0].detach().cpu()
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


class EvaluationDataset():
    '''
    Expected init parameters:
        l: Length of strings (excluding SOS/EOS)
        eval_dataset: Type of dataset (see below)
        eval_type: Eval type is either 'full' or 'random'
        n_samples: number of samples to take if eval_type='random'
        
    The class holds:
        self.full_data: full dataset of eval_dataset type 
        self.sampled_data: n_samples random samples (without repetition) from full_data
        self.data: data to use, self.full_data if eval_type if full, otherwise is self.sampled_data
    
    Three types of datasets are available.
    All prompts are autoregressive prompts, prepened with SOS 
    -- limited:
        Contains l samples. l is the maximum string length (exlusing SOS/EOS) seen during training. 
        Consider 1 <= l0 <= l / 2. For each, add inputs:
        000...0 (l0 zeros) and 
        000...01 (l0 zeros and one '1')
    -- randomised
        Contains 100 samples.
        Consider 8 <= l0 <= 32. For each, make 4 samples of l1, s.t. 1 <= l1 <= l0:
        000...011..1 (l0 zeros and l1 ones)
    -- complete
        All sequences of length 64 that can be completed according to the grammar
    '''
    
    def __init__(self, l, eval_dataset, eval_type='full', n_samples=100, T=None, sampling_eps=None, device=None):
        self.l = l
        self.eval_dataset = eval_dataset
        self.eval_type = eval_type
        self.n_samples = n_samples
        self.T = T
        self.sampling_eps = sampling_eps
        self.device = device
        
        self.full_data = []
        if eval_dataset == 'limited':
            self._init_limited()
        elif eval_dataset == 'randomised':
            self._init_randomised()
        elif eval_dataset == 'complete':
            self._init_complete()
        elif eval_dataset == 'diffusion':
            self._init_diffusion()
        elif eval_dataset == 'unconditional':
            self._init_unconditional()
        
        self.sampled_data = self.full_data.clone()[torch.randperm(self.full_data.shape[0])][:n_samples]
        self.data = self.full_data.clone() if eval_type == 'full' else self.sampled_data
         
    def _init_limited(self):
        '''
        For each l0 in [1, l//2], we add two sequences: 
        000...0 (l0 zeros) and 
        000...01 (l0 zeros and one '1')
        
        Total samples: l//2 (l0 values) * 2 (sequences per l0) = l samples
        '''
        for l0 in range(1, self.l // 2 + 1):
            self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [MASK_token]*(self.l + 1 - l0)).unsqueeze(0))
            self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [1] + [MASK_token]*(self.l - l0)).unsqueeze(0))
        
        self.full_data = torch.cat(self.full_data, dim=0)
        
    def _init_randomised(self):
        '''
        For each l0 (# of zeros) in [8, 32], we sample 4 values of l1 (# of ones) in [1, l0], 
        and add the corresponding sequence.
        '''
        for l0 in range(8, 33):
            # range [1, l0]
            sampled_l1 = torch.randperm(l0)[:4] + 1 
            for l1 in sampled_l1:
                self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [1]*l1 + [MASK_token] * (self.l + 1 - l0 - l1)).unsqueeze(0))
                
        self.full_data = torch.cat(self.full_data, dim=0)
        
    def _init_complete(self):
        '''
        For each l0 in [32, 64], we add the sequence with l0 zeros and l1 ones, where l1 = 64 - l0.
        
        Total samples: 33 (l0 values) * 34 / 2 = 561
        '''
        
        for l0 in range(32, 65):
            for l1 in range(0, 64-l0+1):
                self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [1]*l1 + [MASK_token] * (self.l + 1 - l0 - l1)).unsqueeze(0))
        
        self.full_data = torch.cat(self.full_data, dim=0)
    
    def _init_diffusion(self):
        grammar = anbnGrammar(self.l)
        grammar.generate_seq()  # generates the data and stores in grammar.data
        grammar.data = grammar.data[torch.randperm(grammar.data.shape[0])]
        dataset = Dataset(
            grammar.data, 
            self.device,
            self.T,
            self.sampling_eps
            )
        fixed_dataset = get_fixed_dataset(dataset, self.device, batch_size=self.l//2)

        self.full_data = fixed_dataset[0][0]
        
    def _init_unconditional(self):
        '''
        Fully masked sequences. 
        
        500 samples.
        
        '''
        
        self.full_data = torch.concat([torch.full((500, 1), SOS_token).long(), torch.full((500, self.l + 1), MASK_token).long()], dim=1)


def evaluation_loss(model, dataloader, device):
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_token).to(device)
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            B, L = X_batch.shape
            
            logits = model(X_batch)
            loss = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            mask = (X_batch==MASK_token).view(B*L).float()
            loss *= mask
            loss = torch.sum(loss) / torch.sum(mask)
                
            total_loss += loss.item()
            # predictions = torch.argmax(logits, dim=-1)
            # print("Predictions:", predictions)
            # print("Ground Truth:", y_batch)
            
    print(f'Evaluation, Loss: {total_loss/len(dataloader)}')

def get_timeline(max_len=258, steps=None, idx=-1):
    ans = [(-1, -1)] * max_len 
    
    for t, step in enumerate(steps):
        for idx, val in enumerate(step):
            if val != MASK_token:
                if ans[idx] == (-1, -1):
                    ans[idx] = (t, val.item())
    
    y_pred = steps[-1].clone()
    s, mid_start, pad_start = '', False, False
    s += "IDX " + str(idx) + " " + ''.join([str(i) for i in y_pred.tolist()]) + '\n'
    cnt_zeros, cnt_ones = (y_pred==0).sum(), (y_pred==1).sum()
    s += f' zeros={cnt_zeros}, ones={cnt_ones} \n'
    
    for idx, i in enumerate(ans):
        s += f'{idx:>4}: {i[1]:<20} set at {i[0]:>10}\n'
        if idx + 1 < len(ans) and ans[idx + 1][1] == 1 and mid_start == False:
            s += '------MIDDLE------\n'
            mid_start = True
        elif idx + 1 < len(ans) and ans[idx + 1][1] == PAD_token and pad_start == False:
            s += '------START_OF_PADDING------\n'
            pad_start = True
    
    return ans, s

# eval_type: diffusion or autoregressive
# samples_type for anbn: random or full
def evaluation_from_generation(model, 
                               grammar, 
                               evaluation_dataset=None, 
                               T=500, 
                               strategy = 'categorical', 
                               temperature=1.0, 
                               write_steps=False, 
                               device='cpu', 
                               figures_path=None, 
                               loss_log_path=None, 
                               output_path=None, 
                               save_mode=False, 
                               denoise="0",
                               gaussian_noise=False,
                               sigma=1.0,
                               cutoff=None,
                               investigate=False,
                               n_first_tokens=10**9,
                               attention_every=10,
                               max_attention_tokens=24):
    # r1, r2, both, format
    stats = np.array([0, 0, 0, 0])
    stats_eos = np.array([0, 0, 0, 0])  # stats for sequences that contain EOS
    total = 0
    total_eos = 0  # count of sequences containing EOS
    sequences = []
    sequences_eos = []  # sequences that contain EOS

    # Optionally prepare output file if saving is enabled.
    if save_mode:
        with open(output_path, "w") as f:
            f.write("")

    print(f"Evaluation on data, shape: f{evaluation_dataset.data.shape}")
    _is_oracle_model = model.oracle if hasattr(model, 'oracle') else False
    _oracle_for_eval = None if _is_oracle_model else oracleModel(vocab_size=model.vocab_size, device=device)
    unmaskModel = ScheduledUnmasker(model, device, T=T, denoise=denoise,
                                    oracle=_is_oracle_model, oracle_model=_oracle_for_eval, gaussian_noise=gaussian_noise, sigma=sigma)
        
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(tqdm(evaluation_dataset.data)):
            total += 1
            
            if model.architecture == 'diffusion':
                if write_steps == False:
                    y_pred = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s)), strategy, temperature=temperature) # no batch dimension
                else:
                    y_pred, steps, timesteps_log, error_message = unmaskModel(s, ((s == MASK_token).sum() / torch.numel(s)), strategy, temperature=temperature, return_steps=True)
            else:
                y_pred = get_prediction(model, s, max_tokens=cutoff)  # autoregressive generation; no batch dimension

            # `grammar.evaluate()` uses Python loops/indexing; it's much faster on CPU tensors
            # Moving a single (L,) tensor to CPU is cheap compared to thousands of tiny GPU syncs
            y_pred_cpu = y_pred.detach().to("cpu")
            y_pred_stats = grammar.evaluate(y_pred_cpu)
            stats += y_pred_stats
            seq_str = ''.join([str(i) for i in y_pred_cpu.tolist()])
            sequences.append(seq_str)

            # Track finished sequences (those containing EOS)
            has_eos = (y_pred_cpu == EOS_token).any().item()
            if has_eos:
                stats_eos += y_pred_stats
                total_eos += 1
                sequences_eos.append(seq_str)

            if save_mode:
                if (y_pred_stats == 0).sum() != 0:
                    with open(output_path, 'a') as f:
                        f.write("IDX " + str(idx) + " " + seq_str)
                        is_format_ok = ('True' if y_pred_stats[-1] == 1 else 'False')
                        cnt_zeros, cnt_ones = (y_pred_cpu==0).sum(), (y_pred_cpu==1).sum()
                        f.write(f' zeros={cnt_zeros}, ones={cnt_ones}, format={is_format_ok} \n')
                    
                        if write_steps == True:
                            f.write('Full denoising log: \n')
                            prev = torch.tensor([0])
                            for step in steps:
                                if step.tolist() != prev.tolist():
                                    f.write(''.join([str(i) for i in step.tolist()]) + '\n')
                                    cnt_zeros, cnt_ones, masks = (step==0).sum(), (step==1).sum(), (step==MASK_token).sum()
                                    f.write(f' zeros={cnt_zeros}, ones={cnt_ones}, masks={masks} \n')
                                    prev = step
                            
                            if error_message is not None:
                                f.write(f"Oracle error message: {error_message}\n")
                            f.write('-' * 30 + '\n')
                            
            if save_mode and write_steps == True:
                with open(figures_path / f'IDX={idx}.txt', 'a') as f:
                    ans, output_str = get_timeline(max_len=grammar.l+2, steps=steps, idx=idx)  
                    f.write(output_str)
                    
                    if error_message is not None:
                        f.write(f"Oracle error message: {error_message}\n")
                
                # HTML report: comprehensive (investigate=True) or basic (failing only)
                if investigate:
                    html_path = figures_path / f'IDX={idx}.html'
                    is_failing = (y_pred_stats == 0).sum() != 0
                    case_label = 'FAILING' if is_failing else 'CORRECT'

                    attention_hooks = None
                    attention_msg = None
                    try:
                        attention_hooks = attach_attention_hooks(model)
                    except ValueError as e:
                        attention_msg = str(e)

                    kl_divergences = []
                    timestep_values = []

                    with open(html_path, 'w') as hf:
                        hf.write("""<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Investigation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 16px; }
        .meta { background:#f6f8fa; padding:10px; border-radius:8px; margin-bottom:12px; }
        .step { border:1px solid #ddd; border-radius:8px; padding:12px; margin:12px 0; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; word-wrap: break-word; }
        img { max-width: 100%; height: auto; border:1px solid #ccc; border-radius:6px; }
    </style>
</head>
<body>
""")
                        hf.write(f"<h2>{case_label} Case IDX={idx}</h2>\n")
                        hf.write("<div class='meta mono'>\n")
                        hf.write(f"Input: {html.escape(str(s.tolist()))}<br/>\n")
                        hf.write(f"Final: {html.escape(str(y_pred_cpu.tolist()))}<br/>\n")
                        hf.write(f"Eval (r1, r2, both, fmt): {html.escape(str(y_pred_stats.tolist()))}<br/>\n")
                        hf.write(f"Zeros: {(y_pred_cpu==0).sum()}, Ones: {(y_pred_cpu==1).sum()}<br/>\n")
                        if attention_msg is not None:
                            hf.write(f"Attention maps unavailable: {html.escape(attention_msg)}<br/>\n")
                        else:
                            hf.write(f"Attention snapshots every {attention_every} step(s); max {max_attention_tokens} tokens.<br/>\n")
                        hf.write("</div>\n")

                        # -- Denoising log --
                        hf.write("<h3>Denoising Log</h3>\n<pre class='mono'>\n")
                        prev = torch.tensor([0])
                        for step in steps:
                            if step.tolist() != prev.tolist():
                                hf.write(''.join([str(i) for i in step.tolist()]) + '\n')
                                cnt_z, cnt_o, cnt_m = (step==0).sum().item(), (step==1).sum().item(), (step==MASK_token).sum().item()
                                hf.write(f' zeros={cnt_z}, ones={cnt_o}, masks={cnt_m}\n')
                                prev = step
                        hf.write("</pre>\n")

                        # -- Oracle error --
                        if error_message is not None:
                            hf.write(f"<h3>Oracle Error</h3>\n<pre class='mono'>{html.escape(error_message)}</pre>\n")

                        # -- Per-step KL investigation --
                        hf.write("<h3>Per-Step Investigation</h3>\n")
                        for step_idx, seq_step in enumerate(steps):
                            if step_idx == len(steps) - 1:
                                continue
                            timestep = timesteps_log[step_idx].detach().clone() if isinstance(timesteps_log[step_idx], torch.Tensor) else torch.tensor(timesteps_log[step_idx], device=device)
                            if step_idx > 0 and (seq_step == MASK_token).sum() == (steps[step_idx - 1] == MASK_token).sum():
                                continue

                            model_input = seq_step.unsqueeze(0) if seq_step.dim() == 1 else seq_step
                            timestep_input = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep

                            predicted_logits = model(model_input, timestep_input).squeeze(0)
                            predicted_probs = F.softmax(predicted_logits, dim=-1)

                            dt_seq = seq_step if seq_step.dim() == 1 else seq_step.squeeze(0)
                            expected_result = determineTokenDistribution(dt_seq, vocab_size=model.vocab_size, device=device)

                            if expected_result[0] is None:
                                hf.write(f"<div class='step mono'>No valid completion after step {step_idx}. Stopping.</div>\n")
                                break

                            exp_probs = expected_result[1][:n_first_tokens].to(device)
                            eps = 1e-8
                            kl_per_pos = (exp_probs * (torch.log(exp_probs + eps) - torch.log(predicted_probs + eps))).sum(dim=-1)
                            pred_conf = predicted_probs.max(dim=-1).values
                            pred_argmax = predicted_probs.argmax(dim=-1)
                            exp_argmax = exp_probs.argmax(dim=-1)
                            H_exp = -(exp_probs * torch.log(exp_probs + eps)).sum(dim=-1)
                            H_pred = -(predicted_probs * torch.log(predicted_probs + eps)).sum(dim=-1)

                            div = kl_per_pos.mean().item()
                            kl_divergences.append(div)
                            timestep_values.append(timestep.item())

                            masked_pos = (dt_seq == MASK_token)
                            low_conf = pred_conf < 0.6
                            argmax_mismatch = pred_argmax != exp_argmax
                            flagged = torch.where(masked_pos & (low_conf | argmax_mismatch))[0]

                            hf.write("<div class='step'><div class='mono'>\n")
                            hf.write(f"Step {step_idx} | Timestep: {timestep.item():.4f} | KL: {div:.4f}<br/>\n")
                            hf.write(f"Seq: {html.escape(str(dt_seq.tolist()))}<br/>\n")
                            if step_idx + 1 < len(steps):
                                hf.write(f"Next: {html.escape(str(steps[step_idx + 1].tolist()))}<br/><br/>\n")

                            for pos in flagged.tolist():
                                kl = kl_per_pos[pos].item()
                                he = H_exp[pos].item()
                                hp = H_pred[pos].item()
                                pc = pred_conf[pos].item()
                                exp_row = ', '.join(f'{x:.4f}' for x in exp_probs[pos][:6].tolist())
                                pred_row = ', '.join(f'{x:.4f}' for x in predicted_probs[pos][:6].tolist())
                                hf.write(f"Pos {pos} (tok={dt_seq[pos].item()}) KL={kl:.4f} H_exp={he:.4f} H_pred={hp:.4f} Conf={pc:.4f}<br/>\n")
                                hf.write(f"&nbsp;&nbsp;Expected:&nbsp;&nbsp;[{exp_row}]<br/>\n")
                                hf.write(f"&nbsp;&nbsp;Predicted: [{pred_row}]<br/>\n")

                            should_render_attention = (
                                attention_hooks is not None
                                and flagged.numel() > 0
                                and (step_idx % max(1, attention_every) == 0)
                            )
                            if should_render_attention:
                                attn_maps = extract_attention_maps(model, dt_seq, device=device, timestep=timestep_input)
                                b64 = _attention_grid_to_base64(attn_maps, dt_seq, max_tokens=max_attention_tokens)
                                if b64 is not None:
                                    hf.write(f"<h4>Attention maps</h4>\n<img src='data:image/png;base64,{b64}' />\n")

                            hf.write("</div></div>\n")

                        # -- Inline KL plot --
                        if kl_divergences:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(timestep_values, kl_divergences, marker='o', linewidth=1.5, markersize=4)
                            ax.set_xlabel('Timestep')
                            ax.set_ylabel('KL Divergence')
                            ax.set_title(f'KL Divergence over Denoising — IDX={idx} ({case_label})')
                            ax.invert_xaxis()
                            fig.tight_layout()
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            buf.seek(0)
                            kl_b64 = base64.b64encode(buf.read()).decode('utf-8')
                            hf.write(f"<h3>KL Divergence Plot</h3>\n<img src='data:image/png;base64,{kl_b64}' />\n")

                        hf.write("</body></html>\n")

                    if attention_hooks is not None:
                        remove_hooks(attention_hooks)

                elif (y_pred_stats == 0).sum() != 0:
                    html_path = figures_path / f'IDX={idx}.html'
                    with open(html_path, 'w') as f:
                        f.write("""<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Failing Case Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 16px; }
        .meta { background:#f6f8fa; padding:10px; border-radius:8px; margin-bottom:12px; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
""")
                        f.write(f"<h2>Failing Case IDX={idx}</h2>\n")
                        f.write("<div class='meta mono'>\n")
                        f.write(f"Final sequence: {y_pred_cpu.tolist()}<br/>\n")
                        f.write(f"Evaluation (r1, r2, both, fmt): {y_pred_stats.tolist()}<br/>\n")
                        f.write(f"Zeros: {(y_pred_cpu==0).sum()}, Ones: {(y_pred_cpu==1).sum()}, Format: {y_pred_stats[-1]}\n")
                        f.write("</div>\n")
                        f.write("<h3>Denoising Log</h3>\n")
                        f.write("<pre class='mono'>\n")
                        prev = torch.tensor([0])
                        for step in steps:
                            if step.tolist() != prev.tolist():
                                f.write(''.join([str(i) for i in step.tolist()]) + '\n')
                                cnt_zeros, cnt_ones, masks = (step==0).sum(), (step==1).sum(), (step==MASK_token).sum()
                                f.write(f' zeros={cnt_zeros}, ones={cnt_ones}, masks={masks}\n')
                                prev = step
                        f.write("</pre>\n")
                        if error_message is not None:
                            f.write("<h3>Oracle Error</h3>\n")
                            f.write(f"<pre class='mono'>{error_message}</pre>\n")
                        f.write("</body></html>\n")
    
    eos_denom = max(total_eos, 1)
    evaluation_log = f"""
    Evaluation from generation satisfies rule #1: {stats[0]}/{total} ({stats[0]/total})
    Evaluation from generation satisfies rule #2: {stats[1]}/{total} ({stats[1]/total})
    Evaluation from generation satisfies both rules: {stats[2]}/{total} ({stats[2]/total})
    Evaluation from generation satisfies satisfies format: {stats[3]}/{total} ({stats[3]/total})
    Finished sequences (with EOS): {total_eos}/{total}
    [Finished only] satisfies rule #1: {stats_eos[0]}/{total_eos} ({stats_eos[0]/eos_denom})
    [Finished only] satisfies rule #2: {stats_eos[1]}/{total_eos} ({stats_eos[1]/eos_denom})
    [Finished only] satisfies both rules: {stats_eos[2]}/{total_eos} ({stats_eos[2]/eos_denom})
    [Finished only] satisfies format: {stats_eos[3]}/{total_eos} ({stats_eos[3]/eos_denom})
    """

    # Always print the summary.
    print(evaluation_log)

    # Optionally append the summary to a log file if saving is enabled.
    if save_mode and loss_log_path is not None:
        with open(loss_log_path, "a") as f:
            f.write(evaluation_log + "\n")

    return stats / total, stats_eos / eos_denom, total_eos, sequences, sequences_eos
  
