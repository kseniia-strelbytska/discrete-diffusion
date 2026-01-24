import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import yaml
from types import SimpleNamespace
from datetime import datetime
from loss import rblb
from noise_schedule_unmask import ScheduledUnmasker
from evaluation_tools import evaluation_loss, evaluation_from_generation
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

def setup_experiment_dirs(experiment_path='new_diffusion/'):
    model_path = MODELS_DIR / experiment_path
    figure_path = FIGURES_DIR / experiment_path
    loss_log_path = figure_path / 'loss_log.txt'
    output_path = figure_path / 'outputs.txt'
    
    model_path.mkdir(parents=True, exist_ok=False)
    figure_path.mkdir(parents=True, exist_ok=False)
    
    config_src = PROJECT_ROOT / "src" / "config.yaml"
    config_dst = model_path / "config.yaml"
    shutil.copy2(config_src, config_dst)
    
    print(f'Setup finished: directory {experiment_path}')
    
    dirs = SimpleNamespace(
    model_path=model_path,
    figure_path=figure_path,
    loss_log_path=loss_log_path,
    output_path=output_path)
    
    return dirs

class TransformerClassifier(torch.nn.Module):
    def __init__(self, max_len=16, vocab_size=6, n_head=4, n_layers=2, embed_dim=128, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.l = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
         # Transformer/encoder layer
        self.layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
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
    def __init__(self, y, device='cpu'):
        self.y = y.to(device)
        self.device = device
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        y_sample = self.y[index]
        prob = torch.rand((1, ), device=self.device) # prob of having a mask (ie the timestep)
        mask = torch.rand_like(y_sample, dtype=torch.float, device=self.device) < prob.item()
        X_sample = torch.where(mask == True, torch.full_like(y_sample, torch.tensor(MASK_token, device=self.device)), y_sample)
        
        return X_sample, y_sample, prob

def get_fixed_dataset(dataset, batch_size=32):
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
def train(model, T=500, eos_weight=1.0, dirs=None, evaluation_config = None, epochs=5, lr=1e-3, weight_decay=0.01, train_dataloader=None, test_dataset=None, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = rblb(eos_weight=eos_weight, device=device)
    
    stats = [[], [], [], [], []] # r1, r2, both, format, epochsteps
    test_loss_stats = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    with open(dirs.loss_log_path, 'a') as f:
        f.write('-'*20 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        #cs = []
        
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
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        with open(dirs.loss_log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}\n")
        
        model.eval()
        test_loss = 0
        for X_batch, y_batch, timestep in test_dataset:
            logits = model(X_batch)
            loss = loss_fn(X_batch, logits, y_batch, timestep)
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Average Test Loss: {avg_test_loss:.4f}")
        with open(dirs.loss_log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Average Test Loss: {avg_test_loss:.4f}\n")
            
        test_loss_stats.append(avg_test_loss)
        ax1.clear()
        ax1.plot(np.arange(1, epoch+2), test_loss_stats)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Test Loss (fixed dataset)')
        ax1.set_title('Test Loss vs Epoch')
        ax1.grid(True)
            
        if (epoch + 1) % evaluation_config.eval_every == 0:
            new_stats = evaluation_from_generation(model, 
                                                   grammar, 
           
                                                   data=None, 
                                                   T=T, 
                                                   eval_type=evaluation_config.eval_type, 
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

if __name__ == '__main__':
    cfg = load_config('./config.yaml')
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJECT_ROOT / cfg.paths.models_dir
    FIGURES_DIR = PROJECT_ROOT / cfg.paths.figures_dir
    experiment_name = cfg.paths.experiment_name
    experiment_path_dated = experiment_name + f'_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
    dirs = setup_experiment_dirs(experiment_path_dated)
        
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Device configuration
    if cfg.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else device)
    else:
        device = torch.device(cfg.device)
    print(f'Using device: {device}')
    
    if cfg.data.grammar == 'anbn':
        grammar = anbnGrammar(cfg.data.l)
    else:
        grammar = initialGrammar(cfg.data.l)
    
    grammar.data = grammar.generate_seq()
    dataset = Dataset(grammar.data, device=device)        
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [cfg.data.train_split, 1 - cfg.data.train_split])
    print(f'Dataset len: {len(dataset)}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
    # fixed test dataset
    fixed_test_dataset = get_fixed_dataset(test_dataset, batch_size=cfg.data.batch_size)

    model = TransformerClassifier(
        max_len=cfg.model.max_len,
        vocab_size=cfg.model.vocab_size,
        n_head=cfg.model.n_head,
        n_layers=cfg.model.n_layers,
        embed_dim=cfg.model.embed_dim,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout)
    
    model = model.to(device)
    # model.load_state_dict(torch.load('./models/anbn_diffusion_v8/diffusion_epochs=1'))
    model = train(model=model, 
                  T=cfg.model.T,
                  eos_weight=cfg.model.eos_weight,
                  dirs=dirs,
                  evaluation_config=cfg.evaluation,
                  epochs=cfg.training.epochs, 
                  lr=cfg.training.learning_rate,
                  weight_decay=cfg.training.weight_decay,
                  train_dataloader=train_dataloader, 
                  test_dataset=fixed_test_dataset,
                  device=device
                  )
    # torch.save(model.state_dict(), f'./models/anbn_diffusion_v5/diffusion_epochs=5000')