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

## Project Structure

```
discrete-diffusion/
├── configs/               # All experiment configs (YAML)
│   ├── config.yaml        # Main config (anbn grammar, RPE model)
│   ├── AR_config.yaml     # Autoregressive baseline config
│   └── config_*.yaml      # Per-architecture / sweep configs
├── src/
│   ├── datasets/          # Grammar definitions and datasets
│   │   ├── anbn.py        # a^n b^n grammar
│   │   ├── initialgrammar.py
│   │   ├── re_grammar.py  # All RE grammars (wrapper)
│   │   ├── re_data.py     # RE data generators
│   │   ├── dataset.py     # Masking dataset for diffusion training
│   │   ├── evaluation_dataset.py  # Evaluation prompt sets
│   │   └── constants.py   # Token IDs (EOS, SOS, PAD, MASK)
│   ├── models/            # Model architectures
│   │   ├── model.py       # Classic transformer
│   │   ├── model_RPE.py   # Relative Positional Encoding
│   │   ├── model_T5.py    # T5-style RPE
│   │   └── ...
│   ├── oracle/            # Oracle model and oracle evaluation scripts
│   │   ├── deterministic_token_distribution.py
│   │   ├── eval_oracle_sweep.py
│   │   └── ...
│   ├── eval_scripts/      # Standalone evaluation scripts
│   │   ├── eval_sweep_checkpoints.py
│   │   └── investigate_token_distribution.py
│   ├── schedules/         # Noise schedules
│   ├── main.py            # Training / eval entry point
│   └── trainer.py
├── models/                # Saved model checkpoints
├── figures/               # Training plots and logs
└── results/               # Evaluation results
```

---

## Running experiments

Edit a config in `configs/` to specify your experiment parameters.

```bash
# Train (default config)
python src/main.py --config configs/config.yaml --mode train --save

# Evaluate a checkpoint
python src/main.py --config configs/config.yaml --mode eval

# Investigate token distributions
python src/main.py --config configs/config_investigation.yaml --mode investigate
```

| Argument      | Type   | Default                    | Description                          |
|---------------|--------|----------------------------|--------------------------------------|
| `--config`    | `str`  | `../configs/config.yaml`   | Path to the configuration file       |
| `--mode`      | `str`  | `train`                    | `train`, `eval`, or `investigate`    |
| `--save`      | flag   | `False`                    | Save model checkpoints and figures   |
| `--verbose`   | flag   | `False`                    | Enable verbose output                |
| `--schedule`  | `str`  | *(from config)*            | Override noise schedule: `categorical` or `gaussian` |
| `--sigma`     | float  | *(from config)*            | Override Gaussian sigma              |

---

## Grammar Selection

Set `data.grammar` in your config to choose the training grammar. All grammars use token IDs: **SOS=3, EOS=2, PAD=4, MASK=5**, with grammar content tokens starting at 0.

### Built-in grammars

| `data.grammar` | Description | `model.vocab_size` |
|----------------|-------------|-------------------|
| `anbn` | a^n b^n — equal counts, a's before b's | 6 |
| `initial` | Equal 0s and 1s, no alternating substrings | 6 |

### Formal grammar library (RE grammars)

These grammars come from the rule-extrapolation benchmark. The grammar rule is evaluated as a single pass/fail check.

| `data.grammar` | Description | `model.vocab_size` |
|----------------|-------------|-------------------|
| `aNbN` | Same as `anbn` (a^n b^n via RE generators) | 6 |
| `abN` | Equal a's and b's, any order | 6 |
| `baN` | Begins with b, even number of a's | 6 |
| `bbaN` | b's before a's, even number of a's | 6 |
| `aNbM` | a's before b's, any counts | 6 |
| `aNbNaN` | a^N b^N a^N pattern | 6 |
| `aNbNcN` | a^N b^N c^N — three equal-count sections | 7 |
| `brackets` | Matched `[]` pairs | 8 |
| `parentheses` | Matched `()` pairs | 6 |
| `parentheses_and_brackets` | Nested matched `()` and `[]` | 8 |
| `separated_parentheses_and_brackets` | Either `()` only or `[]` only per sequence | 8 |
| `not_nested_parentheses_and_brackets` | Non-nested matched `()` and `[]` | 8 |

**Example config snippet for `aNbNcN`:**
```yaml
data:
  grammar: aNbNcN
  l: 60          # sequence content length; total seqs: l//3
model:
  vocab_size: 7  # must match grammar's vocab_size above
  max_len: 62    # l + 2
```

**Example for matched parentheses and brackets:**
```yaml
data:
  grammar: parentheses_and_brackets
  l: 64
model:
  vocab_size: 8
  max_len: 66
```

### Evaluation dataset types

For the `anbn` / `aNbN` grammar, structured prompt sets are available:

| `evaluation.eval_dataset` | Description |
|---------------------------|-------------|
| `limited` | In-distribution prompts (up to l/2 a's + some b's) |
| `randomised` | Random in-distribution prompts (default) |
| `complete` | Out-of-distribution prompts (longer than training) |
| `diffusion` | Noisy samples from training distribution |
| `unconditional` | Fully masked sequences — works with **any grammar** |

For RE grammars other than `aNbN`, use `eval_dataset: unconditional`.

---

## Config Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed for reproducibility |
| `device` | `auto` \| `cpu` \| `cuda` \| `mps` | Device to run training on |
| **data** | | |
| `data.l` | int | Max content length (sequences padded to `l+2`) |
| `data.train_split` | float | Fraction of data used for training |
| `data.batch_size` | int | Training batch size |
| `data.grammar` | str | Grammar type (see table above) |
| **model** | | Transformer model parameters |
| `model.architecture` | str | `classic`, `RPE`, `RPE_KQ`, `T5`, `FIRE`, `v2`, `autoregressive`, `RE`, `timestep`, `oracle` |
| `model.max_len` | int | Set to `l + 2` |
| `model.vocab_size` | int | Must match grammar (see table above) |
| `model.n_head` | int | Number of attention heads |
| `model.n_layers` | int | Number of transformer layers |
| `model.embed_dim` | int | Embedding dimension |
| `model.dim_feedforward` | int | Feed-forward layer dimension |
| `model.dropout` | float | Dropout rate |
| `model.T` | int | Number of denoising timesteps |
| `model.eos_weight` | float | EOS token loss weight (compensates for scarcity) |
| **training** | | |
| `training.epochs` | int | Number of training epochs |
| `training.learning_rate` | float | Learning rate |
| `training.loss_type` | `eq8` \| `eq9` | Loss variant |
| **schedule** | | Noise schedule |
| `schedule.type` | `categorical` \| `gaussian` | Schedule type (or use `--schedule` CLI flag) |
| `schedule.sigma` | float | Gaussian sigma (if using gaussian schedule) |
| **evaluation** | | |
| `evaluation.eval_every` | int | Evaluate every N epochs |
| `evaluation.n_samples` | int | Number of evaluation prompts |
| `evaluation.eval_dataset` | str | Prompt type (see table above) |
| `evaluation.eval_type` | `full` \| `random` | Use all prompts or a random subset |
| **paths** | | |
| `paths.models_dir` | str | Checkpoint directory (default `models`) |
| `paths.figures_dir` | str | Figures directory (default `figures`) |
| `paths.experiment_name` | str | Subdirectory name for this run |
| `notes` | str | Free-text experiment description |

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