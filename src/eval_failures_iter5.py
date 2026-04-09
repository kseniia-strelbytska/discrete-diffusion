"""
Find which (l0, l1) pairs fail in the complete eval dataset for iter5 epoch 20k.
Run from src/ directory.
"""
import sys
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset
from noise_schedule_unmask import ScheduledUnmasker
from constants import EOS_token, SOS_token, PAD_token, MASK_token
from model_RPE import RPETransformerClassifier


def dict_to_ns(d):
    return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = PROJECT_ROOT / "models" / "grammar-suffix-rpe-iter5_09042026_141834"
    CONFIG_PATH = MODEL_DIR / "config.yaml"

    with open(CONFIG_PATH) as f:
        cfg = dict_to_ns(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grammar = anbnGrammar(cfg.data.l)
    grammar.generate_seq()

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

    # Load epoch 20k EMA checkpoint
    ema_weights = torch.load(MODEL_DIR / "ema_epochs=20000", map_location=device)
    model.load_state_dict(ema_weights)
    model.eval()

    # Full evaluation dataset (all 561 samples)
    eval_dataset = EvaluationDataset(
        l=cfg.data.l,
        eval_dataset="complete",
        eval_type="random",
        n_samples=561,
        T=cfg.model.T,
        sampling_eps=cfg.model.sampling_eps,
        device=device,
    )
    eval_dataset.data = eval_dataset.data.to(device)

    unmasker = ScheduledUnmasker(model, device=device, T=cfg.model.T, denoise="ltr")

    failures = []
    successes = []

    print("Evaluating all 561 sequences (iter5 epoch 20k)...")
    for sample in eval_dataset.data:
        toks_raw = sample.tolist()
        l0 = sum(1 for t in toks_raw if t == 0)
        l1 = 0
        for t in toks_raw:
            if t == 1:
                l1 += 1
            elif t == MASK_token:
                break

        n_masked = (sample == MASK_token).sum().item()
        timestep = n_masked / len(sample)

        with torch.no_grad():
            output = unmasker(sample, timestep, strategy='None', temperature=0.0)

        output_cpu = output.detach().cpu()
        result = grammar.evaluate(output_cpu)

        if result[0] == 1:
            successes.append((l0, l1))
        else:
            toks = output_cpu.tolist()
            try:
                eos_idx = toks.index(EOS_token)
            except ValueError:
                eos_idx = len(toks)
            ones_start_idx = l0 + 1
            gen_ones = [t for t in toks[ones_start_idx:eos_idx] if t != PAD_token and t != MASK_token]
            failures.append({
                'l0': l0, 'l1': l1,
                'expected': l0,
                'got': len(gen_ones),
                'diff': len(gen_ones) - l0,
                'body': l0 + len(gen_ones),
            })

    print(f"\n=== Failure Analysis ({len(failures)}/561 failures) ===")
    print(f"Accuracy: {(561-len(failures))/561:.4f} ({561-len(failures)}/561)")

    print(f"\n=== Failures by l0 (n zeros) ===")
    by_l0 = defaultdict(list)
    for f in failures:
        by_l0[f['l0']].append(f)

    for l0 in sorted(by_l0.keys()):
        fs = by_l0[l0]
        print(f"  l0={l0:2d}: {len(fs)} failures")
        for f in fs:
            print(f"    l1={f['l1']:2d}: expected {f['expected']} ones, got {f['got']} (diff={f['diff']:+d})")

    print(f"\n=== Success rate by l0 ===")
    by_l0_total = defaultdict(int)
    for l0 in range(32, 65):
        by_l0_total[l0] = 64 - l0 + 1
    for l0 in sorted(set(s[0] for s in successes) | set(f['l0'] for f in failures)):
        total = by_l0_total[l0]
        failed = len(by_l0.get(l0, []))
        print(f"  l0={l0:2d}: {total-failed}/{total} correct")


if __name__ == "__main__":
    main()
