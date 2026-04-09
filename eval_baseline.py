"""
Baseline evaluation script: loads the best RPE model and evaluates on the complete dataset.
Run from the repo root: python eval_baseline.py
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from model_RPE import RPETransformerClassifier
from constants import MASK_token

# ── Config ──────────────────────────────────────────────────────────────────
L          = 256
MAX_LEN    = 258      # L + 2
VOCAB_SIZE = 6
N_HEAD     = 4
N_LAYERS   = 4
EMBED_DIM  = 256
FF_DIM     = 1024
DROPOUT    = 0.1
LN_EPS     = 2e-4
T          = 500
SAMPLING_EPS = 1e-5
TEMPERATURE  = 0.1
DENOISE      = "0"   # current default (random)

CHECKPOINT = Path("models/RPE-architecture-fixed_23032026_204407/model_epochs=85000")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model ────────────────────────────────────────────────────────────────────
model = RPETransformerClassifier(
    max_len=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    embed_dim=EMBED_DIM,
    dim_feedforward=FF_DIM,
    dropout=DROPOUT,
    layer_norm_eps=LN_EPS,
    sampling_eps=SAMPLING_EPS,
).to(device)

state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Grammar & Evaluation Dataset ─────────────────────────────────────────────
grammar = anbnGrammar(L)
grammar.generate_seq()

for eval_name in ["complete", "randomised"]:
    print(f"\n{'='*60}")
    print(f"Evaluating on dataset: {eval_name}")
    eval_dataset = EvaluationDataset(
        l=L,
        eval_dataset=eval_name,
        eval_type="full",
        n_samples=200,
        T=T,
        sampling_eps=SAMPLING_EPS,
        device=device,
    )
    eval_dataset.data = eval_dataset.data.to(device)
    print(f"Eval samples: {len(eval_dataset.data)}")

    stats, stats_eos, total_eos, seqs, seqs_eos = evaluation_from_generation(
        model,
        grammar,
        evaluation_dataset=eval_dataset,
        T=T,
        strategy=None,
        temperature=TEMPERATURE,
        write_steps=False,
        device=device,
        figures_path=Path("figures"),
        loss_log_path=None,
        output_path=None,
        save_mode=False,
        denoise=DENOISE,
        cutoff=MAX_LEN,
    )
    print(f"Rule1: {stats[0]:.4f} | Rule2: {stats[1]:.4f} | Both: {stats[2]:.4f} | Format: {stats[3]:.4f}")
    print(f"Finished (EOS) seqs: {total_eos}/{len(eval_dataset.data)}")

# Write baseline report
report = f"""BASELINE REPORT — RPE Architecture (best checkpoint: epoch 85000)
=================================================================
Architecture   : RPE (Relative Positional Encoding Transformer)
Layers         : {N_LAYERS}
Embed dim      : {EMBED_DIM}
Heads          : {N_HEAD}
FFN dim        : {FF_DIM}
Max seq len    : {MAX_LEN}  (L={L})
Diffusion T    : {T}
Denoising mode : {DENOISE} (random order)
Temperature    : {TEMPERATURE}
Checkpoint     : {CHECKPOINT}

NOTE: The RPE model ignores the timestep parameter entirely.
      It operates as a masked language model (BERT-style).

See stdout for per-dataset accuracy numbers above.
"""
with open("baseline_report.txt", "w") as f:
    f.write(report)
print("\nbaseline_report.txt written.")
