## Project Overview

An ongoing investigation into **rule extrapolation** and **length generalisation** in **Discrete Diffusion Models** on out-of-distribution formal language prompts.

This work is inspired by recent advances in structured generative modelling, particularly:

- **Sequence Modelling with Discrete Diffusion**  
  https://arxiv.org/pdf/2406.07524

- **Compositional Generalisation on OOD Prompts in Language Models** (NeurIPS 2024)  
  https://proceedings.neurips.cc/paper_files/paper/2024/file/3d9ef68629089da055334c2d41dfcf93-Paper-Conference.pdf

---

## Core Objectives

1. Train **transformer-based discrete diffusion models** on formal languages defined as the **intersection of two rules**, and analyse whether and how these rules are learned.
2. Evaluate **out-of-distribution generalisation** on prompts that violate one rule, and compare performance against **autoregressive transformer baselines**.
3. Investigate **length generalisation** by testing model performance on sequences longer than those seen during training.

---

## Motivation

Generalisability is a key challenge in modern generative modelling. Understanding whether models can **extrapolate abstract rules**, rather than merely interpolate training distributions, is critical for robust reasoning and compositional generalisation in neural sequence models.

## Evaluation Metrics Used

### 1. Diffusion

- `n_samples=100`; sampled from training data
- Randomly masked with random `0.8 <= masking_probability <= 1.0`

### 2. Autoregressive

- `n_samples=100`; sampled from the dataset:
  - For all `1 <= l <= 128`, add `['0' x l]` and `['0' x l + '1']` to the dataset
  - e.g. `000` and `0001` for `l=3`

### Accuracy Metrics

Both metrics track accuracy on:
- **Rule 1**: the number of 0s and 1s match
- **Rule 2**: all 0s precede all 1s
- **Both Rules**: strings that satisfy both Rule 1 and Rule 2
- **Format** (in some versions): the string follows the order `SOS[0s][1s]EOS[PAD]`, with 1 SOS, 1 EOS; zero or more 0s, 1s, PAD

## Running experiments

Change src/config.yaml to specify parameters of your experiment. An example config.yaml is in src/ 

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed for reproducibility |
| `device` | `auto` \| `cpu` \| `gpu` \| `mps` | Device to run training on |
| **data** |  |  |
| `data.l` | int | Max length of the data |
| `data.train_split` | float | Fraction of data samples that go to training data |
| `data.batch_size` | int | Batch size for training and testing data |
| `data.grammar` | `anbn` \| `initial` | Grammar type (see `src/anbn.py` or `src/initialgrammar.py`) |
| **model** |  | Transformer model parameters |
| `model.max_len` | int | Set to `l + 2` (max length + EOS/SOS) |
| `model.vocab_size` | int | Vocabulary size (default 6) |
| `model.n_head` | int | Number of attention heads |
| `model.n_layers` | int | Number of transformer layers |
| `model.embed_dim` | int | Embedding dimension for each token |
| `model.dim_feedforward` | int | Dimensionality of linear layer in each encoder layer |
| `model.dropout` | float | Dropout rate |
| `model.T` | int | Number of denoising steps during unmasking process (see `src/noise_schedule_unmask.py`) |
| `model.eos_weight` | float | Weight of the EOS token class (due to scarcity of EOS tokens) |
| **training** |  |  |
| `training.epochs` | int | Number of training epochs |
| `training.learning_rate` | float | Learning rate |
| **evaluation** |  |  |
| `evaluation.eval_every` | int | Run evaluation every X epochs |
| `evaluation.n_samples` | int | Number of samples to run evaluation on |
| `evaluation.samples_type` | `random` \| `full` | Sampling type (see `src/evaluation_tools/*.py`) |
| `evaluation.eval_type` | `autoregressive` \| `diffusion` | Type of evaluation prompts (see `src/evaluation_tools/*.py`) |
| **paths** |  |  |
| `paths.models_dir` | str | Directory for saving models (default `models`) |
| `paths.figures_dir` | str | Directory for saving figures (default `figures`) |
| `paths.experiment_name` | str | Directory name for this experiment |
| `notes` | str | Description of your experiment |

# Usage

Run with:
```bash
$ python src/main.py
```

## Expected Results
```
outputs/
└── [experiment_name]/
    ├── models/
    │   ├── config_snapshot.yaml
    │   └── diffusion_epochs_500.pt
    └── figures/
        ├── plot.png
        └── loss_log.txt
```

### models/[experiment_name]

Contains:
- Model weights, saved every `evaluation.eval_every` epochs
- A copy of the yaml config file used in this experiment

### figures/[experiment_name]

Contains:
- `plot.png`: A graph of accuracy metrics tracked (accuracy on rule 1, rule 2, both rules, and format). Expect to see a line graph after `2*evaluation.eval_every` epochs
- `loss_log.txt`: A file containing a log of loss outputs, test loss outputs, and evaluation accuracy outputs (i.e. a copy of the terminal output)