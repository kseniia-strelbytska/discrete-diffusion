import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers.optimization import get_inverse_sqrt_schedule

from anbn import anbnGrammar
from constants import EOS_token, SOS_token, PAD_token, MASK_token
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from initialgrammar import initialGrammar
from loss import rblb
from noise_schedule_unmask import ScheduledUnmasker

def dict_to_ns(d):
    return SimpleNamespace(**{
        k: dict_to_ns(v) if isinstance(v, dict) else v
        for k, v in d.items()
    })

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict_to_ns(cfg)

def setup_experiment_dirs(PROJECT_ROOT, MODELS_DIR, FIGURES_DIR, config_path, experiment_path='new_diffusion/'):
    model_path = MODELS_DIR / experiment_path
    figure_path = FIGURES_DIR / experiment_path
    loss_log_path = figure_path / 'loss_log.txt'
    output_path = figure_path / 'outputs.txt'
    
    model_path.mkdir(parents=True, exist_ok=False)
    figure_path.mkdir(parents=True, exist_ok=False)
    
    config_dst = model_path / "config.yaml"
    shutil.copy2(config_path, config_dst)
    
    print(f'Setup finished: directory {experiment_path}')
    
    dirs = SimpleNamespace(
    model_path=model_path,
    figure_path=figure_path,
    loss_log_path=loss_log_path,
    output_path=output_path)
    
    return dirs

class TransformerClassifier(torch.nn.Module):
    def __init__(self, max_len=16, vocab_size=6, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1, layer_norm_eps=2e-4):
        super().__init__()

        self.l = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
         # Transformer/encoder layer
        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                nhead=n_head, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout,
                                                layer_norm_eps=layer_norm_eps, 
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        # Predictor head: a simple linear layer
        self.fc = nn.Linear(embed_dim, vocab_size - 1) # do not allow mask (5) prediction 
        
        PE = torch.zeros((max_len, embed_dim))
        pos = torch.arange(max_len).unsqueeze(-1)
        div = torch.pow(1e4, 2 * torch.arange(0, embed_dim // 2) / embed_dim)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)
        
        self.register_buffer('PE', PE)

    def forward(self,
                X: torch.Tensor):
        B, L = X.shape
        X = self.embedding(X) # (B, L, E) = (128, 20, 10)     
        E = X.shape[-1]
        
        ## Sinusoidal positional encoding 
        X += self.PE[:L, :].unsqueeze(0)

        # Pass through network
        X = self.transformer_encoder(src=X)
        X = self.fc(X)

        return X

class Dataset(torch.utils.data.Dataset):
    def __init__(self, y, inverse_t=False, device='cpu'):
        # if invers_t=True, mask with probability sampled from 1/x
        self.y = y.to(device)
        self.device = device
        self.inverse_t = inverse_t
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        y_sample = self.y[index]
        
        if not self.inverse_t:
            prob = torch.rand((1, ), device=self.device) # prob of having a mask (ie the timestep)
        else:
            prob = self.sample_inverse_t()
        
        mask = torch.rand_like(y_sample, dtype=torch.float, device=self.device) < prob.item()
        X_sample = torch.where(mask == True, torch.full_like(y_sample, torch.tensor(MASK_token, device=self.device)), y_sample)
        
        return X_sample, y_sample, prob
    
    def sample_inverse_t(self):
        CLIP_VALUE = 1e-2
        u = np.random.uniform(0, 1)
        log_clip = np.log(CLIP_VALUE)
        sampled_val = np.exp(u * (np.log(1.0) - log_clip) + log_clip)
        
        return torch.tensor(sampled_val)

def get_fixed_dataset(dataset, batch_size=32, device='cpu'):
    fixed_dataset = []
    batch_size = 32
    for idx in range(len(dataset)):
        X, y, timestep = dataset.__getitem__(idx)
        X = X.to(device)
        y = y.to(device)
        timestep = timestep.to(device)
        
        if not fixed_dataset or fixed_dataset[-1][0].shape[0] == batch_size:
            fixed_dataset.append((X.unsqueeze(0), y.unsqueeze(0), timestep.unsqueeze(0)))
        else:
            fixed_dataset[-1] = (torch.cat([fixed_dataset[-1][0], X.unsqueeze(0)], dim=0), 
                               torch.cat([fixed_dataset[-1][1], y.unsqueeze(0)], dim=0), 
                               torch.cat([fixed_dataset[-1][2], timestep.unsqueeze(0)], dim=0))
            
    return fixed_dataset

# noise_resolution -- T in the scheduled unmasker
def train(model, grammar, T=500, eos_weight=1.0, inverse_t=False, dirs=None, evaluation_config=None, epochs=5, lr=1e-3, num_warmup_steps=1000, weight_decay=0.01, train_dataloader=None, test_dataset=None, evaluation_dataset=None, device='cpu', verbose=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if num_warmup_steps != 0:
        lr_scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=num_warmup_steps)
    loss_fn = rblb(eos_weight=eos_weight, inverse_t=inverse_t, device=device)
    
    stats = [[], [], [], [], []] # r1, r2, both, format, epochsteps
    test_loss_stats, train_loss_stats = [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    with open(dirs.loss_log_path, 'a') as f:
        f.write('-'*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    model.train()
    
    epochs_iter = range(epochs) if verbose else tqdm(range(epochs), desc="Training Epochs")
    for epoch in epochs_iter:
        total_loss = 0

        model.train()
        for X_batch, y_batch, timestep in train_dataloader:
            # Ensure batches are on the correct device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            timestep = timestep.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(X_batch, logits, y_batch, timestep)
            loss.backward()
            optimizer.step()
            if num_warmup_steps != 0:
                lr_scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        train_loss_stats.append(avg_loss)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, timestep in test_dataset:
                # Ensure batches are on the correct device
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                timestep = timestep.to(device)
            
                logits = model(X_batch)
                loss = loss_fn(X_batch, logits, y_batch, timestep)
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_dataset)
            test_loss_stats.append(avg_test_loss)
        
        current_lr = lr 
        if num_warmup_steps != 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Current lr: {current_lr}")
        with open(dirs.loss_log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Current lr: {current_lr}\n")
            
        ax1.clear()
        ax1.plot(np.arange(1, epoch+2), train_loss_stats)
        ax1.plot(np.arange(1, epoch+2), test_loss_stats)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(['Train Loss', 'Test Loss (fixed dataset)'], loc="lower right")
        ax1.set_title('Loss vs Epoch')
        ax1.grid(True)
            
        if (epoch + 1) % evaluation_config.eval_every == 0:
            new_stats = evaluation_from_generation(model, 
                                                   grammar, 
                                                   evaluation_dataset=evaluation_dataset,
                                                   T=T, 
                                                   device=device, 
                                                   loss_log_path=dirs.loss_log_path,
                                                   output_path=dirs.output_path)
            for i in range(4):
                stats[i].append(new_stats[i]) 
            stats[-1].append(epoch + 1)
            
            ax2.clear()
            ax2.plot(stats[-1], stats[0])
            ax2.plot(stats[-1], stats[1])
            ax2.plot(stats[-1], stats[2])
            ax2.plot(stats[-1], stats[3])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend(["Rule 1", "Rule 2", "Both Rules", "Format"], loc="lower right")
            torch.save(model.state_dict(), dirs.model_path / f'model_epochs={epoch + 1}')
            
        plt.tight_layout()
        plt.savefig(dirs.figure_path / 'plot.png', dpi=150)
        
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="discrete diffusion training and evaluation")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the configuration file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval.')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJECT_ROOT / cfg.paths.models_dir
    FIGURES_DIR = PROJECT_ROOT / cfg.paths.figures_dir
    experiment_name = cfg.paths.experiment_name
    experiment_path_dated = experiment_name + f'_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
    dirs = setup_experiment_dirs(PROJECT_ROOT, MODELS_DIR, FIGURES_DIR, args.config, experiment_path_dated)
        
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Device configuration
    device = None
    if cfg.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(cfg.device)
    print(f'Using device: {device}')
    
    if cfg.data.grammar == 'anbn':
        grammar = anbnGrammar(cfg.data.l)
    else:
        grammar = initialGrammar(cfg.data.l)
    
    grammar.data = grammar.generate_seq()
    dataset = Dataset(grammar.data, inverse_t=cfg.model.inverse_t, device=device)

    print(f'Dataset len: {len(dataset)} using inverse_t sampling {cfg.model.inverse_t}')  
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [cfg.data.train_split, 1 - cfg.data.train_split])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    fixed_test_dataset = get_fixed_dataset(test_dataset, batch_size=cfg.data.batch_size, device=device) # fixed test dataset
    
    evaluation_dataset = EvaluationDataset(l=cfg.data.l,
                                          eval_dataset=cfg.evaluation.eval_dataset,
                                          eval_type=cfg.evaluation.eval_type,
                                          n_samples=cfg.evaluation.n_samples)
    print(f'Evaluation Dataset len: {len(evaluation_dataset.data)}')

    model = TransformerClassifier(
        max_len=cfg.model.max_len,
        vocab_size=cfg.model.vocab_size,
        n_head=cfg.model.n_head,
        n_layers=cfg.model.n_layers,
        embed_dim=cfg.model.embed_dim,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        layer_norm_eps=cfg.model.layer_norm_eps)
    model = model.to(device)
    
    if args.mode == 'train':
        model = train(model, 
                    grammar,
                    T=cfg.model.T,
                    eos_weight=cfg.model.eos_weight,
                    inverse_t=cfg.model.inverse_t,
                    dirs=dirs,
                    evaluation_config=cfg.evaluation,
                    epochs=cfg.training.epochs, 
                    lr=cfg.training.learning_rate,
                    num_warmup_steps=cfg.training.num_warmup_steps,
                    weight_decay=cfg.training.weight_decay,
                    train_dataloader=train_dataloader, 
                    test_dataset=fixed_test_dataset,
                    evaluation_dataset=evaluation_dataset,
                    device=device,
                    verbose=args.verbose
                    )
    else:    
        model.load_state_dict(torch.load(MODELS_DIR / 'n_embed=128_ff=1024_drop=0.1_27012026_221030/model_epochs=96500', map_location=torch.device('cpu')))
        
        for iter_eval_dataset in ['complete', 'randomised', 'limited', ]:
            print(f'Evaluation dataset: {iter_eval_dataset}')
            current_evaluation_dataset = EvaluationDataset(l=cfg.data.l,
                                          eval_dataset=iter_eval_dataset,
                                          eval_type=cfg.evaluation.eval_type,
                                          n_samples=cfg.evaluation.n_samples)
            evals = evaluation_from_generation(model, 
                                                grammar, 
                                                evaluation_dataset=current_evaluation_dataset,
                                                T=cfg.model.T, 
                                                write_steps=True,
                                                device=device, 
                                                figures_path=dirs.figure_path,
                                                loss_log_path=dirs.loss_log_path,
                                                output_path=dirs.output_path)
            
        exit(0)
        
        # test different seeds
        # model.load_state_dict(torch.load(MODELS_DIR / f'anbn_diffusion_v8/diffusion_epochs={32500}'))
        # unmask = ScheduledUnmasker(model, T=1500, device=device)
        # l = 59
        # input_X = grammar.data[l-1].clone()
        # input_X[l+2:] = MASK_token 
        # output_X, steps = unmask(input_X, ((input_X == MASK_token).sum() / torch.numel(input_X)), return_steps=True)
        # line, output_str = get_timeline(max_len=grammar.l+2, steps=steps)

        # seeds = [i for i in range(1, 11)]
        
        # for chosen_seed in seeds:
        #     torch.manual_seed(chosen_seed)
        #     random.seed(chosen_seed)
        #     np.random.seed(chosen_seed)
            
        #     experiment_path_dated = f'{experiment_name}_seed={chosen_seed}_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
        #     dirs = setup_experiment_dirs(PROJECT_ROOT, MODELS_DIR, FIGURES_DIR, args.config, experiment_path_dated)
            
        #     new_stats = evaluation_from_generation(model, 
        #                                                 grammar, 
        #                                                 evaluation_dataset=evaluation_dataset,
        #                                                 T=cfg.model.T, 
        #                                                 write_steps=False,
        #                                                 device=device, 
        #                                                 figures_path=dirs.figure_path,
        #                                                 loss_log_path=dirs.loss_log_path,
        #                                                 output_path=dirs.output_path)
        
        # exit(0)
        
        # re-evaluate and plot accuracies of pre-trained models
        stats = [[], [], [], [], []] # r1, r2, both, format, epochsteps

        for i in range(27, 28):
            epochs = 32500
            model.load_state_dict(torch.load(MODELS_DIR / f'anbn_diffusion_v8/diffusion_epochs={epochs}'))

            new_stats = evaluation_from_generation(model, 
                                                    grammar, 
                                                    evaluation_dataset=evaluation_dataset,
                                                    data=None, 
                                                    T=500, 
                                                    device=device, 
                                                    loss_log_path=dirs.loss_log_path,
                                                    output_path=dirs.output_path)
            for i in range(4):
                stats[i].append(new_stats[i]) 
            stats[-1].append(epochs)
        
        plt.clf()
        fig, ax = plt.subplots()             
        ax.plot(stats[-1], stats[0])
        ax.plot(stats[-1], stats[1])
        ax.plot(stats[-1], stats[2])
        ax.plot(stats[-1], stats[3])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend(["Rule 1", "Rule 2", "Both Rules", "Format"], loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'v8_32500epochs_test_run.png', dpi=150)
        
        exit(0)

if __name__ == '__main__':
    main()