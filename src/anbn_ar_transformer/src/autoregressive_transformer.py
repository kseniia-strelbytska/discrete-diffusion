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

from evaluation_tools import evaluation_loss, evaluation_from_generation
from generation_and_predictions import get_prediction
from anbn import anbnGrammar
from initialgrammar import initialGrammar
from constants import EOS_token, SOS_token, PAD_token, MASK_token

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

class Model(nn.Module):
    def __init__(self, max_len=20, vocab_size=5, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
        super().__init__() 

        self.n_head=n_head
        self.n_layers=n_layers
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, X):
        B, L = X.shape 
        mask = torch.triu(torch.ones(L, L, device=X.device), diagonal=1).bool()
        padding_mask = (X == PAD_token)
    
        positions = self.positional_embedding(torch.arange(0, L, device=X.device).unsqueeze(0)) # (1, L) -> (1, L, E)
        X = self.embedding(X) + positions # (B, L, E)

        X  = self.transformer_encoder(src=X, mask=mask, is_causal=True) # apply mask to make it a unidirectional block!
        X = self.fc(X)

        return X

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device='cpu'):
        super().__init__()
        
        self.X = X.to(device)
        self.y = y.to(device)
        self.device = device
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.X.shape[0]
    
    

def train(model, grammar, eos_weight=1.0, dirs=None, evaluation_config=None, lr_scheduler=False, epochs=5, lr=1e-3, train_dataloader=None, test_dataloader=None, device='cpu', verbose=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if lr_scheduler:
        scheduler = get_inverse_sqrt_schedule(
            optimizer=optimizer,
            num_warmup_steps=1000
        )
    
    class_weights = torch.ones(model.vocab_size, device=device)
    class_weights[EOS_token] = eos_weight
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    stats = [[], [], [], [], []] #r1, r2, both, format, epochsteps
    test_loss_stats, train_loss_stats = [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    with open(dirs.loss_log_path, 'a') as f:
        f.write('-'*20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataloader:
            B, L = X_batch.shape
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss_raw = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            mask = (y_batch.view(B*L) != PAD_token).float()
            loss = (loss_raw * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            
            if lr_scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        train_loss_stats.append(avg_loss)
        
        model.eval()
        test_loss = 0
        for X_batch, y_batch in test_dataloader:
            B, L = X_batch.shape
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss_raw = loss_fn(logits.view(B*L, -1), y_batch.view(B*L))
            mask = (y_batch.view(B*L) != PAD_token).float()
            loss = (loss_raw * mask).sum() / mask.sum()
            
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_dataloader)
        test_loss_stats.append(avg_test_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {'No schedule' if not lr_scheduler else scheduler.get_last_lr()[0]}")
        with open(dirs.loss_log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {'No schedule' if not lr_scheduler else scheduler.get_last_lr()[0]}\n")
            
        ax1.clear()
        ax1.plot(np.arange(1, epoch+2), train_loss_stats)
        ax1.plot(np.arange(1, epoch+2), test_loss_stats)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(['Train Loss', 'Test Loss'], loc="lower right")
        ax1.set_title('Loss vs Epoch')
        ax1.grid(True)
        
        if (epoch + 1) % evaluation_config.eval_every == 0:
            new_stats = evaluation_from_generation(model=model, 
                                                   grammar=grammar, 
                                                   data=None, 
                                                   eval_type=grammar.default_eval_type, 
                                                   samples_type=evaluation_config.samples_type, 
                                                   n_samples=evaluation_config.n_samples,
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
    parser = argparse.ArgumentParser(description="AR transformer training and evaluation")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the configuration file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    return parser.parse_args()


def set_seed(seed):
    """Comprehensive seed setting for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms (may impact performance)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJECT_ROOT / cfg.paths.models_dir
    FIGURES_DIR = PROJECT_ROOT / cfg.paths.figures_dir
    experiment_name = cfg.paths.experiment_name
    experiment_path_dated = experiment_name + f'_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
    dirs = setup_experiment_dirs(PROJECT_ROOT, MODELS_DIR, FIGURES_DIR, args.config, experiment_path_dated)
    
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
        
    torch.cuda.manual_seed_all(cfg.seed)
    print(f'Using device: {device}')
        
    if cfg.data.grammar == 'anbn':
        grammar = anbnGrammar(cfg.data.l)
    else:
        grammar = initialGrammar(cfg.data.l)
    
    grammar.data = grammar.generate_seq()

    X = grammar.data.clone()[:, :-1].to(device)
    y = grammar.data.clone()[:, 1:].to(device)
    
    dataset = Dataset(X, y, device=device)
    print(f'Dataset len: {len(dataset)}')
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [cfg.data.train_split, 1 - cfg.data.train_split],
        generator=generator
    )
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=0  # Set to 0 for determinism, or keep workers with seed_worker
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # full_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model(max_len=cfg.model.max_len,
        vocab_size=cfg.model.vocab_size,
        n_head=cfg.model.n_head,
        n_layers=cfg.model.n_layers,
        embed_dim=cfg.model.embed_dim,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout) 
    model = model.to(device)
    # model.load_state_dict(torch.load('./models/anbn_trained_models/rule2_autoregressive_transformer_epochs=1500'))
    model = train(model=model, 
                  grammar=grammar,
                  eos_weight=cfg.model.eos_weight,
                  dirs=dirs,
                  evaluation_config=cfg.evaluation,
                  lr_scheduler=cfg.training.lr_scheduler,
                  epochs=cfg.training.epochs, 
                  lr=cfg.training.learning_rate,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  device=device,
                  verbose=args.verbose)
    # torch.save(model.state_dict(), f'./rule2_autoregressive_transformer_500')
                
    # evaluation_loss(model, test_dataloader)
    # new_stats = evaluation_from_generation(model, grammar, data=None, eval_type=grammar.default_eval_type, samples_type='full', n_samples=100)

if __name__ == '__main__':
    main()