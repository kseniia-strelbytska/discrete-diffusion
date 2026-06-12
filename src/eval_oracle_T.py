"""
Evaluate the oracle model across T values on the unconditional dataset.

Standalone (single T):
    python src/eval_oracle_T.py --config config_oracle_T_sweep.yaml --T 100

Standalone (all sweep T values in one run, produces a comparison plot):
    python src/eval_oracle_T.py --config config_oracle_T_sweep.yaml --all-T 10 50 100 200

W&B sweep (one agent = one T value, project/entity come from sweep yaml):
    wandb sweep sweeps/oracle_T_sweep.yaml
    wandb agent <ENTITY/PROJECT/SWEEP_ID>
"""

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anbn import anbnGrammar
from initialgrammar import initialGrammar
from deterministic_token_distribution import oracleModel
from evaluation_tools import EvaluationDataset, evaluation_from_generation
from schedules import CategoricalSchedule, GaussianSchedule


def dict_to_ns(d):
    return SimpleNamespace(**{k: dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg_device):
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(cfg_device)


def get_grammar(grammar_type, l):
    if grammar_type == "anbn":
        return anbnGrammar(l)
    if grammar_type == "initial":
        return initialGrammar(l)
    raise ValueError(f"Unknown grammar: {grammar_type!r}")


def get_schedule(cfg):
    schedule_cfg = getattr(cfg, "schedule", None)
    schedule_type = getattr(schedule_cfg, "type", "categorical")
    sigma = getattr(schedule_cfg, "sigma", 1.0)
    if schedule_type == "gaussian":
        return GaussianSchedule(sigma=sigma)
    return CategoricalSchedule()


def eval_one_T(cfg, device, grammar, schedule):
    """Run evaluation for a single T value (cfg.model.T). Returns a metrics dict."""
    model = oracleModel(vocab_size=cfg.model.vocab_size, device=device)

    dataset = EvaluationDataset(
        l=cfg.data.l,
        eval_dataset="unconditional",
        eval_type="full",
        n_samples=500,
        T=cfg.model.T,
        sampling_eps=cfg.model.sampling_eps,
        device=device,
    )

    strategy = cfg.strategy or "categorical"
    denoise = getattr(getattr(cfg, "training", None), "denoise", "0")

    stats, stats_eos, total_eos, sequences, _ = evaluation_from_generation(
        model,
        grammar,
        evaluation_dataset=dataset,
        T=cfg.model.T,
        strategy=strategy,
        temperature=cfg.temperature,
        device=device,
        schedule=schedule,
        gaussian_noise=isinstance(schedule, GaussianSchedule),
        sigma=schedule.sigma if isinstance(schedule, GaussianSchedule) else 1.0,
        cutoff=cfg.evaluation.cutoff,
        denoise=denoise,
    )

    finished_pct = total_eos / len(sequences) if sequences else 0.0
    return {
        "T": cfg.model.T,
        "Rule_Accuracy/rule1_acc": float(stats[0]),
        "Rule_Accuracy/rule2_acc": float(stats[1]),
        "Rule_Accuracy/both_rules_acc": float(stats[2]),
        "Rule_Accuracy/format_acc": float(stats[3]),
        "Rule_Accuracy_Finished/rule1_acc": float(stats_eos[0]),
        "Rule_Accuracy_Finished/rule2_acc": float(stats_eos[1]),
        "Rule_Accuracy_Finished/both_rules_acc": float(stats_eos[2]),
        "Rule_Accuracy_Finished/format_acc": float(stats_eos[3]),
        "finished_pct": finished_pct,
        "total_eos": int(total_eos),
        "n_sequences": len(sequences),
    }


def plot_metrics(all_metrics):
    """Bar chart of rule accuracies for each T value. Returns a wandb.Image."""
    T_values = [m["T"] for m in all_metrics]
    metric_keys = [
        "Rule_Accuracy/rule1_acc",
        "Rule_Accuracy/rule2_acc",
        "Rule_Accuracy/both_rules_acc",
        "Rule_Accuracy/format_acc",
    ]
    labels = ["Rule 1", "Rule 2", "Both rules", "Format"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(T_values))
    bar_width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, prefix, title in [
        (axes[0], "Rule_Accuracy/", "All sequences"),
        (axes[1], "Rule_Accuracy_Finished/", "Finished sequences (EOS present)"),
    ]:
        for i, (key_suffix, label, color) in enumerate(
            zip(["rule1_acc", "rule2_acc", "both_rules_acc", "format_acc"], labels, colors)
        ):
            key = prefix + key_suffix
            values = [m[key] for m in all_metrics]
            ax.bar(x + i * bar_width, values, bar_width, label=label, color=color)

        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels([f"T={t}" for t in T_values])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Finished % overlay on right axis of axes[1]
    ax2 = axes[1].twinx()
    finished_pcts = [m["finished_pct"] for m in all_metrics]
    ax2.plot(x + 1.5 * bar_width, finished_pcts, "k--o", label="Finished %", linewidth=1.5)
    ax2.set_ylabel("Finished %")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Oracle accuracy vs. denoising steps T (unconditional dataset)", fontsize=12)
    fig.tight_layout()

    return wandb.Image(fig)


def main():
    parser = argparse.ArgumentParser(description="Oracle T-sweep evaluation")
    parser.add_argument("--config", default="config_oracle_T_sweep.yaml")
    parser.add_argument("--T", type=int, default=None, help="Override model.T for a single run")
    parser.add_argument(
        "--all-T",
        type=int,
        nargs="+",
        metavar="T",
        help="Run all listed T values in one wandb run and produce a comparison plot",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    cfg = dict_to_ns(base_config)

    if args.T is not None:
        cfg.model.T = args.T

    # W&B sweep: one agent = one T value
    in_sweep = os.getenv("WANDB_SWEEP_ID") is not None
    if in_sweep:
        wandb.init()
        for key, value in wandb.config.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            print(f"Sweep override: {key} = {value}")
    elif cfg.wandb.project:
        run_name = (
            f"oracle_T={','.join(str(t) for t in args.all_T)}"
            if args.all_T
            else f"oracle_T={cfg.model.T}"
        )
        wandb.init(
            project=cfg.wandb.project,
            group=cfg.wandb.group or "oracle-T-sweep",
            name=run_name,
            config={"data.l": cfg.data.l, "grammar": cfg.data.grammar},
        )
    else:
        wandb.init(mode="disabled")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = get_device(cfg.device)
    grammar = get_grammar(cfg.data.grammar, cfg.data.l)
    grammar.generate_seq()
    schedule = get_schedule(cfg)

    if args.all_T:
        # Multi-T mode: log each T as a step so W&B draws line charts
        all_metrics = []
        for t_val in args.all_T:
            cfg.model.T = t_val
            print(f"\nRunning T={t_val}…")
            m = eval_one_T(cfg, device, grammar, schedule)
            all_metrics.append(m)
            flat = {k: v for k, v in m.items() if k != "T"}
            wandb.log({"T": t_val, **flat}, step=t_val)
            print(f"  both_rules_acc={m['Rule_Accuracy/both_rules_acc']:.4f}  "
                  f"finished_pct={m['finished_pct']:.4f}")

        wandb.log({"accuracy_vs_T": plot_metrics(all_metrics)})
        plt.close("all")

    else:
        # Single-T mode (sweep agent or --T override)
        print(f"Running oracle evaluation: T={cfg.model.T}, device={device}")
        m = eval_one_T(cfg, device, grammar, schedule)
        wandb.log(m)
        wandb.summary.update(m)
        print(f"\n=== Oracle T={cfg.model.T} Results ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.finish()


if __name__ == "__main__":
    main()
