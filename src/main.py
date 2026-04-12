import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers.optimization import get_inverse_sqrt_schedule
import wandb

from anbn import anbnGrammar
from constants import EOS_token, MASK_token, PAD_token, SOS_token
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from initialgrammar import initialGrammar
from loss import rblb
from noise_schedule_unmask import ScheduledUnmasker
from dataset import Dataset, get_fixed_dataset
from trainer import train
from investigate_token_distribution import investigate_dataset

from model import TransformerClassifier
from model_v2 import v2TransformerClassifier
from model_RPE import RPETransformerClassifier
from model_FIRE import FIRETransformerClassifier
from AR_model_AR import ARTransformerClassifier
from AR_model_RE import TransformerDecoder
from model_timestep import TimestepTransformerClassifier

def dict_to_ns(d):
    return SimpleNamespace(
        **{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()}
    )

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def setup_experiment_dirs(
    PROJECT_ROOT, MODELS_DIR, FIGURES_DIR, config_path, experiment_path="new_diffusion/", save_mode: bool = False
):
    model_path = MODELS_DIR / experiment_path
    figure_path = FIGURES_DIR / experiment_path
    loss_log_path = figure_path / "loss_log.txt"
    output_path = figure_path / "outputs.txt"

    # Only create directories and copy config when saving is enabled.
    if save_mode:
        model_path.mkdir(parents=True, exist_ok=False)
        figure_path.mkdir(parents=True, exist_ok=False)

        config_dst = model_path / "config.yaml"
        shutil.copy2(config_path, config_dst)

        print(f"Setup finished (saving enabled): directory {experiment_path}")

    dirs = SimpleNamespace(
        model_path=model_path,
        figure_path=figure_path,
        loss_log_path=loss_log_path,
        output_path=output_path,
    )

    return dirs

def parse_args():
    parser = argparse.ArgumentParser(
        description="discrete diffusion training and evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--save", action="store_true", help="Enable saving output.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "investigate"],
        help="Mode: train, eval, or investigate.",
    )
    args, unknown = parser.parse_known_args()
    return args

def get_device(cfg_device):
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    else:
        return torch.device(cfg_device)
    return torch.device("cpu")


def get_grammar(cfg_data_grammar, cfg_data_l):
    if cfg_data_grammar == "anbn":
        return anbnGrammar(cfg_data_l)
    if cfg_data_grammar == "initial":
        return initialGrammar(cfg_data_l)

    raise ValueError(f"Invalid grammar type: {cfg_data_grammar}")


def main():
    args = parse_args()
    base_config = load_config(args.config)
    cfg = dict_to_ns(base_config)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJECT_ROOT / cfg.paths.models_dir
    FIGURES_DIR = PROJECT_ROOT / cfg.paths.figures_dir
    experiment_name = cfg.paths.experiment_name
    experiment_path_dated = (
        experiment_name + f'_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
    )
    dirs = setup_experiment_dirs(
        PROJECT_ROOT,
        MODELS_DIR,
        FIGURES_DIR,
        args.config,
        experiment_path_dated,
        save_mode=args.save,
    )

    if os.getenv("WANDB_SWEEP_ID") is not None:
        wandb.init()
        
        if "model_preset" in wandb.config:
            if wandb.config.model_preset == "256_258_256":
                wandb.config.update({
                    "model.l": 256,
                    "model.max_len": 258,
                    "model.lembed_dim": 256,
                    "data.l": 256
                }, allow_val_change=True)
            elif wandb.config.model_preset == "512_518_512":
                wandb.config.update({
                    "model.l": 512,
                    "model.max_len": 518, # Using 518 from your prompt vs 514 in your yaml
                    "model.lembed_dim": 512,
                    "data.l": 512
                }, allow_val_change=True)

        for key, value in wandb.config.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            print(f"Sweep override: {key} = {value}")

    if cfg.wandb.project and cfg.wandb.group:
        wandb.init(project=cfg.wandb.project, group=cfg.wandb.group, config=cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        # CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Device configuration
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    grammar = get_grammar(cfg.data.grammar, cfg.data.l)
    grammar.generate_seq()  # generates the data and stores in grammar.data

    dataset = Dataset(grammar.data, device, cfg.model.T, sampling_eps=cfg.model.sampling_eps, inverse_t=cfg.model.inverse_t)

    print(f"Dataset len: {len(dataset)} using inverse_t sampling {cfg.model.inverse_t}")
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [cfg.data.train_split, 1 - cfg.data.train_split]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, collate_fn=dataset.masking_collate_fn
    )
    
    test_data = dataset.y_data[test_dataset.indices] # grab only test samples
    test_dataset = Dataset(test_data, device, cfg.model.T, 
                       sampling_eps=cfg.model.sampling_eps, inverse_t=cfg.model.inverse_t)
    fixed_test_dataset = get_fixed_dataset(
        test_dataset, device, batch_size=cfg.data.batch_size
    )  # fixed test dataset

    evaluation_dataset = EvaluationDataset(
        l=cfg.data.l,
        eval_dataset=cfg.evaluation.eval_dataset,
        eval_type=cfg.evaluation.eval_type,
        n_samples=cfg.evaluation.n_samples,
        T=cfg.model.T,
        sampling_eps=cfg.model.sampling_eps,
        device=device
    )

    evaluation_dataset.data = evaluation_dataset.data.to(device)
    
    print(f"Evaluation Dataset len: {len(evaluation_dataset.data)}")

    if cfg.model.architecture == "classic":
        model = TransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            sampling_eps=cfg.model.sampling_eps,
        ).to(device)
    elif cfg.model.architecture == "v2":
        model = v2TransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            sampling_eps=cfg.model.sampling_eps,
        ).to(device)
    elif cfg.model.architecture == "RPE":
        model = RPETransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            sampling_eps=cfg.model.sampling_eps,
        ).to(device)
    elif cfg.model.architecture == "FIRE":
        model = FIRETransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            sampling_eps=cfg.model.sampling_eps,
        ).to(device)
    elif cfg.model.architecture == "autoregressive":
        model = ARTransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps
        ).to(device)
    elif cfg.model.architecture == "RE":
        model = TransformerDecoder(
        vocab_size=cfg.model.vocab_size,
        dim_model=cfg.model.embed_dim,
        num_heads=cfg.model.n_head,
        num_decoder_layers=cfg.model.n_layers,
        dropout_p=cfg.model.dropout,
        dim_feedforward=cfg.model.dim_feedforward,
        layer_norm_eps=cfg.model.layer_norm_eps
        ).to(device)
    elif cfg.model.architecture == "timestep":
        model = TimestepTransformerClassifier(
            max_len=cfg.model.max_len,
            vocab_size=cfg.model.vocab_size,
            n_head=cfg.model.n_head,
            n_layers=cfg.model.n_layers,
            embed_dim=cfg.model.embed_dim,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            sampling_eps=cfg.model.sampling_eps
        ).to(device)
    else:
        raise ValueError(f"Invalid model architecture: {cfg.model.architecture}")

    if args.mode == "train":
        model = train(
            model,
            grammar,
            device,
            T=cfg.model.T,
            eos_weight=cfg.model.eos_weight,
            inverse_t=cfg.model.inverse_t,
            dirs=dirs,
            evaluation_config=cfg.evaluation,
            validation_config=cfg.validation,
            epochs=cfg.training.epochs,
            lr=cfg.training.learning_rate,
            num_warmup_steps=cfg.training.num_warmup_steps,
            weight_decay=cfg.training.weight_decay,
            train_dataloader=train_dataloader,
            test_dataset=fixed_test_dataset,
            evaluation_dataset=evaluation_dataset,
            verbose=args.verbose,
            save_mode=args.save,
            strategy=cfg.strategy,
            temperature=cfg.temperature,
            wandb=wandb,
            loss_type=cfg.training.loss_type,
            denoise=cfg.training.denoise,
            cutoff=cfg.evaluation.cutoff
        )
        
        torch.save(
            model.state_dict(), MODELS_DIR / f"model_final_{cfg.model.architecture}.pt"
        )
    elif args.mode == "investigate":
        model.load_state_dict(
            torch.load(
                MODELS_DIR / "RPE-architecture-fixed_23032026_204407/model_epochs=40000", map_location=torch.device("cpu")))
        model = model.to(device)
        
        unmasker = ScheduledUnmasker(model, 
                                     device=device, 
                                     T=cfg.model.T, 
                                     denoise=cfg.training.denoise)
        
        investigate_dataset(model, 
                            unmasker, 
                            device=device, 
                            grammar=grammar, 
                            dataset=evaluation_dataset.data, 
                            figures_dir=dirs.figure_path, 
                            n_first_tokens=cfg.investigation.n_first_tokens)
        print('Finished investigation of token distributions successfully. Check the logs for details.')
    
    elif args.mode == "eval":
        model.load_state_dict(
            # torch.load(
            #     MODELS_DIR
            #     / "n_embed=128_ff=1024_drop=0.1_27012026_221030/model_epochs=96500",
            #     map_location=torch.device("cpu"),
            # )
            
            torch.load(
                PROJECT_ROOT / "models/RPE-architecture-fixed_23032026_204407/model_epochs=40000", map_location=torch.device('cpu')
            )
        )
        
        model = model.to(device)

        for iter_eval_dataset in [
            "limited",
            "randomised",
            "complete",
        ]:
            print(f"Evaluation dataset: {iter_eval_dataset}")
            current_evaluation_dataset = EvaluationDataset(
                l=cfg.data.l,
                eval_dataset=iter_eval_dataset,
                eval_type=cfg.evaluation.eval_type,
                n_samples=cfg.evaluation.n_samples,
                T=cfg.model.T,
                sampling_eps=cfg.model.sampling_eps,
                device=device,
            )
            
            iter_output_path = dirs.output_path.parent / f"outputs_{iter_eval_dataset}.txt"
            iter_figure_path = dirs.figure_path.parent / f"figures_{iter_eval_dataset}"
            iter_figure_path.mkdir(parents=True, exist_ok=True)
            
            evals = evaluation_from_generation(
                model,
                grammar,
                evaluation_dataset=current_evaluation_dataset,
                strategy=cfg.strategy,
                temperature=cfg.temperature,
                T=cfg.model.T,
                write_steps=True,
                device=device,
                figures_path=iter_figure_path,
                loss_log_path=dirs.loss_log_path,
                output_path=iter_output_path,
                save_mode=args.save,
                cutoff=cfg.evaluation.cutoff
            )

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
        stats = [[], [], [], [], []]  # r1, r2, both, format, epochsteps

        for i in range(27, 28):
            epochs = 32500
            model.load_state_dict(
                torch.load(MODELS_DIR / f"anbn_diffusion_v8/diffusion_epochs={epochs}")
            )

            new_stats = evaluation_from_generation(
                model,
                grammar,
                evaluation_dataset=evaluation_dataset,
                data=None,
                T=500,
                device=device,
                loss_log_path=dirs.loss_log_path,
                output_path=dirs.output_path,
            )
            for i in range(4):
                stats[i].append(new_stats[i])
            stats[-1].append(epochs)

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(stats[-1], stats[0])
        ax.plot(stats[-1], stats[1])
        ax.plot(stats[-1], stats[2])
        ax.plot(stats[-1], stats[3])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(["Rule 1", "Rule 2", "Both Rules", "Format"], loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "v8_32500epochs_test_run.png", dpi=150)

        exit(0)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
