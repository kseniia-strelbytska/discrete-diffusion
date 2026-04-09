"""
Evaluate all EMA checkpoints from iter4 on the full 561-sample complete dataset.
Run from src/ directory.
"""
import sys
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from anbn import anbnGrammar
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from model_RPE import RPETransformerClassifier


def dict_to_ns(d):
    return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = PROJECT_ROOT / "models" / "grammar-suffix-rpe-iter4_09042026_121454"
    CONFIG_PATH = MODEL_DIR / "config.yaml"

    with open(CONFIG_PATH) as f:
        cfg = dict_to_ns(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    print(f"Eval dataset size: {len(eval_dataset.data)}")

    checkpoints = [5000, 10000, 15000, 20000, 25000, 30000]
    print(f"\n{'Epoch':>8} | {'Rule1':>6} | {'Rule2':>6} | {'Both':>6} | {'Format':>6}")
    print("-" * 50)

    for epoch in checkpoints:
        ema_path = MODEL_DIR / f"ema_epochs={epoch}"
        ema_weights = torch.load(ema_path, map_location=device)
        model.load_state_dict(ema_weights)
        model.eval()

        stats, stats_eos, total_eos, _, _ = evaluation_from_generation(
            model,
            grammar,
            evaluation_dataset=eval_dataset,
            T=cfg.model.T,
            strategy=cfg.strategy,
            temperature=cfg.temperature,
            write_steps=False,
            device=device,
            figures_path=None,
            loss_log_path=None,
            output_path=None,
            save_mode=False,
            denoise=cfg.training.denoise,
            cutoff=cfg.evaluation.cutoff,
        )

        r1, r2, both, fmt = stats
        n = len(eval_dataset.data)
        print(f"{epoch:>8} | {r1:>6.3f} | {r2:>6.3f} | {both:>6.3f} | {fmt:>6.3f}  ({int(both*n)}/{n})")


if __name__ == "__main__":
    main()
