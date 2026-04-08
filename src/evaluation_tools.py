import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from noise_schedule_unmask import ScheduledUnmasker
from constants import EOS_token, SOS_token, PAD_token, MASK_token
from anbn import anbnGrammar
from dataset import Dataset, get_fixed_dataset
from AR_generation_and_predictions import get_prediction

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
                               cutoff=None):
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
    unmaskModel = ScheduledUnmasker(model, device, T=T, denoise=denoise)
    
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(tqdm(evaluation_dataset.data)):
            total += 1
            
            if model.architecture == 'diffusion':
                if write_steps == False:
                    y_pred = unmaskModel(s, torch.tensor(1.0), strategy, temperature=temperature) # no batch dimension
                else:
                    y_pred, steps = unmaskModel(s, torch.tensor(1.0), strategy, temperature=temperature, return_steps=True)
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
                if y_pred_stats[-1] == 0:
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
                            
                            f.write('-' * 30 + '\n')
                    
            if save_mode and write_steps == True:
                with open(figures_path / f'IDX={idx}.txt', 'a') as f:
                    ans, output_str = get_timeline(max_len=grammar.l+2, steps=steps, idx=idx)  
                    f.write(output_str)
    
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
  
