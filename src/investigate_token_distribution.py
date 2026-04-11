import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from constants import MASK_token, PAD_token, SOS_token, EOS_token
from deterministic_token_distribution import determineTokenDistribution
from model_RPE import RPETransformerClassifier
from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation

def investigate_seq(model, unmasker, device, seq, numeric_log, graphs_log, n_first_tokens=10**9):
    PRECISION = 4
    
    with open(numeric_log, 'a') as numeric_log_file:
        numeric_log_file.write(f'START for sequence: {seq.tolist()}\n')
    
    with torch.no_grad():
        model.eval()
        # Pass a 1D sequence to the unmasker (it accepts both 1D and batched inputs).
        input_seq = torch.tensor(seq).long()
        timestep = ((input_seq == MASK_token).sum() / torch.numel(input_seq))  # proportion of MASK tokens
        final, steps, timesteps = unmasker(input_seq, timestep, return_steps=True)
        prev_timestep = -1
        
        for idx, seq_step in enumerate(steps):
            if idx == len(steps) - 1: # skip the final step where all tokens are unmasked, since the distribution is not meaningful there
                continue 
            # seq_step may be 1D (L,) when unmasker was called with a 1D sequence.
            # The model expects a batched input (B, L), so ensure a batch dim exists.
            timestep = timesteps[idx]
            
            if abs(timestep - prev_timestep) < 1e-6:  # timestep hasn't changed significantly, skip this step
                continue  # skip if timestep hasn't changed (no new tokens unmasked)
            prev_timestep = timestep
            
            model_input = seq_step.unsqueeze(0) if seq_step.dim() == 1 else seq_step
            timestep_input = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep

            predicted_distribution = model(model_input, timestep_input).squeeze(0)  # remove batch dim
            predicted_distribution = torch.softmax(predicted_distribution, dim=-1)[: n_first_tokens]

            # determineTokenDistribution expects a 1D sequence; provide squeezed seq_step
            dt_seq = seq_step if seq_step.dim() == 1 else seq_step.squeeze(0)
            expected_distribution = determineTokenDistribution(dt_seq, 
                                                               vocab_size=model.vocab_size)

            if expected_distribution[0] == None:
                with open(numeric_log, 'a') as numeric_log_file:
                    numeric_log_file.write(f'No valid completion for this sequence exists. Finishing investigation for the case after {idx} step(s).\n')
                break
            else:
                expected_distribution = expected_distribution[1][: n_first_tokens].to(device) # Take only the first n tokens for comparison
                div = F.kl_div(predicted_distribution.log(), expected_distribution, reduction='batchmean').item()

                # round:
                rounded_expected_distribution = [[round(x, PRECISION) for x in row] for row in expected_distribution.tolist()]
                rounded_predicted_distribution = [[round(x, PRECISION) for x in row] for row in predicted_distribution.tolist()]    
                
                with open(numeric_log, 'a') as numeric_log_file:
                    numeric_log_file.write(f'Timestep: {timestep.item():.4f}, KL Divergence: {div:.4f}\n')
                    numeric_log_file.write(f'Expected distribution:\n')
                    for token_idx in range(len(rounded_expected_distribution)):
                        numeric_log_file.write(f'{rounded_expected_distribution[token_idx]}\n')
                        
                    numeric_log_file.write(f'Predicted distribution:\n')
                    for token_idx in range(len(rounded_predicted_distribution)):
                        numeric_log_file.write(f'{rounded_predicted_distribution[token_idx]}\n')

    with open(numeric_log, 'a') as numeric_log_file:
        numeric_log_file.write(f'FINISH for sequence: {seq.tolist()}\n')
        
def investigate_dataset(model, unmasker, device, dataset, figures_dir, n_first_tokens=10**9):
    numeric_log = figures_dir / 'numeric_log.txt'
    graphs_log = figures_dir / 'graphs_log.txt'
    
    with open(figures_dir / 'numeric_log.txt', 'w') as numeric_log_file:
        numeric_log_file.write('') # clear the log file before starting the investigation
    with open(figures_dir / 'graphs_log.txt', 'w') as graphs_log_file:
        graphs_log_file.write('') # clear the log file before starting the investigation
    
    for i in tqdm(range(len(dataset))):
        seq = dataset[i]
        investigate_seq(model, unmasker, device=device, seq=seq, numeric_log=numeric_log, graphs_log=graphs_log, n_first_tokens=n_first_tokens)