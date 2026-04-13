# Iter 5 — Final Experiment Summary

**Result: 561/561 (100%) on the complete evaluation dataset (l0∈[32,64])**  
Stable from epoch 20,000 through 30,000.

---

## How to Reproduce

```bash
cd /workspaces/discrete-diffusion
source .venv/bin/activate
cd src
python main.py --config ../config_grammar_suffix_rpe_iter5.yaml --save
```

Checkpoints and loss logs are written to:
- `models/grammar-suffix-rpe-iter5_<timestamp>/ema_epochs={5000,10000,...,30000}`
- `figures/grammar-suffix-rpe-iter5_<timestamp>/loss_log.txt`

Use the EMA checkpoint at epoch 20,000 or later (all achieve 100%).

---

## What Changed and Why

### 1. Grammar-aware masking (`masking_type: grammar_suffix`)

**Change:** Replace default MDLM random Bernoulli masking with grammar-specific suffix masking.

**How it works:** Every training sample is masked as:
```
[SOS, 0^n, 1^k, MASK^(n-k), MASK_EOS, PAD...]
```
where `k ~ Uniform{0, ..., n}`. The full run of zeros is always revealed; only the ones + EOS region is masked.

**Why:** This exactly matches the evaluation distribution. At eval time, the model always receives `[SOS, 0^n, 1^l1, MASK^(n-l1), ...]` and must predict the remaining ones + EOS. With random masking, the model also trains on masked zeros, which is a distribution the evaluation never presents. Eliminating this mismatch was the single biggest accuracy improvement (from ~0% Rule 1 to ~83% in one step).

**Code:** `src/dataset.py`, `masking_collate_fn`, case `"grammar_suffix"`.

---

### 2. Left-to-right (LTR) denoising at eval

**Change:** Use `denoise: ltr` instead of random-order denoising.

**How it works:** At evaluation, tokens are unmasked strictly left-to-right. The model processes each masked position once, in order, with the timestep (fraction still masked) decreasing after each step.

**Why:** For a^n b^n counting, generating ones left-to-right is the natural inductive structure: the model always has a complete prefix when predicting the next token. Random order denoising asks the model to fill in gaps mid-sequence, which is harder for counting. LTR also runs ~2× faster for the same accuracy.

**Code:** `src/noise_schedule_unmask.py`, `_ltr_forward`.

---

### 3. EMA with slow decay (`ema_decay: 0.9999`)

**Change:** Evaluate using an exponential moving average of weights with decay=0.9999.

**How it works:** A shadow copy of weights is maintained: `shadow = 0.9999 * shadow + 0.0001 * params` after each gradient step. Evaluations always use the shadow weights; gradient updates continue on the live weights.

**Why:** Training the grammar_suffix masking can produce hard batches that cause large gradient spikes, momentarily destabilising the model. Decay=0.9999 is slow enough to smooth these out (the EMA lags ~10,000 gradient steps behind the live model). An earlier run used decay=0.999 (too fast — tracked the spikes).

**Code:** `src/trainer.py`, class `EMA`.

---

### 4. Gradient clipping (`grad_clip_norm: 1.0`)

**Change:** Clip gradient norm to 1.0 before each optimizer step.

**Why:** Grammar-suffix batches that happen to contain many hard cases (large n, few revealed ones) produce outsized gradients. Clipping prevents a single bad batch from overwriting well-learned weights. Combined with slow EMA, this eliminated the 83%→89%→70%→84% oscillation pattern seen in earlier runs.

**Code:** `src/trainer.py`, `torch.nn.utils.clip_grad_norm_`.

---

### 5. Larger model architecture (iter 5 specific)

**Change:** Scale from 6L × 256d × 4H (12M params) to 8L × 512d × 8H (25.5M params).

| Parameter | Iter 3/4 | Iter 5 |
|---|---|---|
| `n_layers` | 6 | 8 |
| `embed_dim` | 256 | 512 |
| `dim_feedforward` | 1024 | 2048 |
| `n_head` | 4 | 8 |

**Why:** Iter 3/4 had one persistent failure: every sequence with n=41 was predicted as n=43 (+2 diff), regardless of how many ones were already revealed. The model placed EOS at absolute position 85 (the n=42 EOS position) for all n=41 inputs — a systematic counting confusion between adjacent values. The larger model's richer representational capacity resolved this; 100% accuracy is stable from epoch 20k onwards.

**Code:** `config_grammar_suffix_rpe_iter5.yaml`, model section.

---

### 6. Lower learning rate and longer warmup for the larger model

**Change:** `learning_rate: 0.0003` (was 0.001), `num_warmup_steps: 3000` (was 1000).

**Why:** The first iter 5 attempt used LR=0.001. The 25.5M parameter model diverged at epoch 1001 (train loss spiked to 59.7 and never recovered). Reducing LR by 3× and tripling the warmup period stabilised training: train loss reached 0.0173 at epoch 1001 and 0.0000 by epoch 2500. A general rule of thumb when scaling model size: LR should scale inversely with the square root of parameter count.

---

## Full Config (`config_grammar_suffix_rpe_iter5.yaml`)

```yaml
seed: 42
device: auto
strategy: None
temperature: 0.0   # greedy decoding

data:
  l: 256
  train_split: 0.9
  batch_size: 64
  grammar: anbn
  masking_type: grammar_suffix      # key: always reveal all zeros

model:
  architecture: RPE
  max_len: 258
  vocab_size: 6
  n_head: 8
  n_layers: 8
  embed_dim: 512
  dim_feedforward: 2048
  dropout: 0.1
  layer_norm_eps: 0.00001
  T: 500
  sampling_eps: 0.00001
  eos_weight: 10.0
  inverse_t: False

training:
  epochs: 30000
  learning_rate: 0.0003            # 3× lower than iter4; required for 25.5M model
  num_warmup_steps: 3000           # 3× longer warmup
  weight_decay: 0.0
  loss_type: eq8
  denoise: ltr                     # left-to-right denoising at eval
  ema_decay: 0.9999                # slow EMA — lags ~10k steps, smooths spikes
  grad_clip_norm: 1.0              # prevents large gradient steps

validation:
  val_every: 500

evaluation:
  eval_every: 5000
  n_samples: 100
  eval_dataset: complete
  eval_type: random
  cutoff: 258

paths:
  models_dir: models
  figures_dir: figures
  experiment_name: grammar-suffix-rpe-iter5
```

---

## Evaluation Checkpoints (Full 561-Sample Dataset)

| Epoch | Rule1 | Rule2 | Both | Count |
|-------|-------|-------|------|-------|
| 5,000  | 0.941 | 1.000 | 0.941 | 528/561 |
| 10,000 | 0.955 | 1.000 | 0.955 | 536/561 |
| 15,000 | 0.955 | 1.000 | 0.955 | 536/561 |
| **20,000** | **1.000** | **1.000** | **1.000** | **561/561** |
| **25,000** | **1.000** | **1.000** | **1.000** | **561/561** |
| **30,000** | **1.000** | **1.000** | **1.000** | **561/561** |

Evaluation uses EMA weights, LTR denoising, greedy decoding (temperature=0).

---

## Loading a Saved Checkpoint

```python
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
from model_RPE import RPETransformerClassifier
from noise_schedule_unmask import ScheduledUnmasker

MODEL_DIR = Path("models/grammar-suffix-rpe-iter5_09042026_141834")

with open(MODEL_DIR / "config.yaml") as f:
    cfg = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v
                             for k, v in yaml.safe_load(f).items()})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

weights = torch.load(MODEL_DIR / "ema_epochs=20000", map_location=device)
model.load_state_dict(weights)
model.eval()

unmasker = ScheduledUnmasker(model, device=device, T=cfg.model.T, denoise="ltr")

# Given a partially-masked sequence `x` (1D tensor, length 258):
# n_masked = (x == MASK_token).sum().item()
# timestep = n_masked / len(x)
# output = unmasker(x, timestep, strategy='None', temperature=0.0)
```

---

## Iteration History (Brief)

| Iter | Key change | 561-sample accuracy |
|------|-----------|---------------------|
| Baseline | Random masking, no EMA | ~91% (randomised set only) |
| 1 | DiT + suffix masking | 0% Rule 1 (masking bug) |
| 2 | RPE + grammar_suffix + EMA 0.999 | 83%→89%→70% (oscillation) |
| 3 | + EMA 0.9999 + grad_clip | 95.7% (24 failures, all n=41) |
| 4 | Larger model, LR=0.001 | Failed to converge |
| **5** | **8L 512d + LR=0.0003** | **100% (epoch 20k–30k)** |
