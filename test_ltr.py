"""
Quick hypothesis test: compare random denoising vs LTR denoising on the
saved RPE model checkpoint, using just 20 sequences from the complete dataset.
Run: python test_ltr.py
"""
import sys, time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset
from noise_schedule_unmask import ScheduledUnmasker
from model_RPE import RPETransformerClassifier
from constants import MASK_token, EOS_token, SOS_token

# ── Config ────────────────────────────────────────────────────────────────
L, MAX_LEN = 256, 258
VOCAB_SIZE, N_HEAD, N_LAYERS = 6, 4, 4
EMBED_DIM, FF_DIM = 256, 1024
T, SAMPLING_EPS, TEMPERATURE = 500, 1e-5, 0.1
CHECKPOINT = Path("models/RPE-architecture-fixed_23032026_204407/model_epochs=85000")
N_TEST = 20  # number of sequences to test

device = torch.device("cpu")

model = RPETransformerClassifier(
    max_len=MAX_LEN, vocab_size=VOCAB_SIZE, n_head=N_HEAD, n_layers=N_LAYERS,
    embed_dim=EMBED_DIM, dim_feedforward=FF_DIM, dropout=0.1,
    layer_norm_eps=2e-4, sampling_eps=SAMPLING_EPS,
).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print(f"Model loaded: {CHECKPOINT}")

grammar = anbnGrammar(L)
grammar.generate_seq()

# Use complete dataset, take first N_TEST sequences
eval_ds = EvaluationDataset(l=L, eval_dataset="complete", eval_type="full",
                             n_samples=N_TEST, T=T, sampling_eps=SAMPLING_EPS, device=device)
test_seqs = eval_ds.full_data[:N_TEST].to(device)
print(f"Testing on {N_TEST} sequences from complete dataset\n")

def run_eval(denoiser_mode, seqs):
    unmasker = ScheduledUnmasker(model, device, T=T, denoise=denoiser_mode)
    results = np.zeros(4)
    t0 = time.time()
    for s in seqs:
        frac_masked = (s == MASK_token).sum() / s.numel()
        y_pred = unmasker(s, frac_masked, temperature=TEMPERATURE)
        results += grammar.evaluate(y_pred.cpu())
    elapsed = time.time() - t0
    results = results / len(seqs)
    print(f"  Mode={denoiser_mode!r:4s} | Rule1={results[0]:.2f} Rule2={results[1]:.2f} "
          f"Both={results[2]:.2f} Format={results[3]:.2f} | {elapsed:.1f}s total")
    return results

print("=== Random denoising (denoise='0') ===")
r_rand = run_eval("0", test_seqs)

print("\n=== Left-to-right denoising (denoise='ltr') ===")
r_ltr = run_eval("ltr", test_seqs)

print(f"\nDelta (LTR - random): Both={r_ltr[2]-r_rand[2]:+.2f}")
if r_ltr[2] > r_rand[2]:
    print("✓ LTR denoising IMPROVES accuracy")
elif r_ltr[2] == r_rand[2]:
    print("= LTR denoising has same accuracy")
else:
    print("✗ LTR denoising hurts accuracy")
