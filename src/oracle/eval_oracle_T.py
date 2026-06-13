"""
Evaluate the oracle model across T values on the unconditional dataset.

Standalone (single T):
    python src/eval_oracle_T.py --config config_oracle_T_sweep.yaml --T 100
    python src/eval_oracle_T.py --config config_oracle_T_sweep.yaml --T 100 --save

Standalone (all sweep T values in one run, produces a comparison plot):
    python src/eval_oracle_T.py --config config_oracle_T_sweep.yaml --all-T 10 50 100 200 --save

W&B sweep (one agent = one T value, project/entity come from sweep yaml):
    wandb sweep sweeps/oracle_T_sweep.yaml
    wandb agent <ENTITY/PROJECT/SWEEP_ID>
"""

import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.grammar_oracles import oracleModel
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
        from datasets.anbn import anbnGrammar
        return anbnGrammar(l)
    if grammar_type == "initial":
        from datasets.initialgrammar import initialGrammar
        return initialGrammar(l)
    from datasets.re_grammar import REGrammar
    if grammar_type in REGrammar.SUPPORTED:
        return REGrammar(grammar_type, l)
    raise ValueError(f"Unknown grammar: {grammar_type!r}")


def get_schedule(cfg):
    schedule_cfg = getattr(cfg, "schedule", None)
    schedule_type = getattr(schedule_cfg, "type", "categorical")
    sigma = getattr(schedule_cfg, "sigma", 1.0)
    if schedule_type == "gaussian":
        return GaussianSchedule(sigma=sigma)
    return CategoricalSchedule()


def setup_dirs(PROJECT_ROOT, cfg, config_path, save_mode):
    """Mirror main.py's setup_experiment_dirs for this script."""
    FIGURES_DIR = PROJECT_ROOT / cfg.paths.figures_dir
    experiment_path = cfg.paths.experiment_name + f'_{datetime.now().strftime("%d%m%Y_%H%M%S")}/'
    figure_path = FIGURES_DIR / cfg.data.grammar / experiment_path
    loss_log_path = figure_path / "loss_log.txt"
    output_path = figure_path / "outputs.txt"
    if save_mode:
        figure_path.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, figure_path / "config.yaml")
        print(f"Saving enabled: {figure_path}")
    return SimpleNamespace(
        figure_path=figure_path,
        loss_log_path=loss_log_path,
        output_path=output_path,
    )


def eval_one_T(cfg, device, grammar, schedule, dirs=None, save_mode=False):
    """Run evaluation for a single T value (cfg.model.T). Returns a metrics dict."""
    model = oracleModel(grammar_name=cfg.data.grammar, vocab_size=cfg.model.vocab_size, device=device)

    dataset = EvaluationDataset(
        l=cfg.data.l,
        eval_dataset="unconditional",
        eval_type=cfg.evaluation.eval_type,
        n_samples=getattr(cfg.evaluation, "n_samples", 500),
        T=cfg.model.T,
        sampling_eps=cfg.model.sampling_eps,
        device=device,
    )

    strategy = cfg.strategy
    denoise = getattr(getattr(cfg, "training", None), "denoise", "0")

    figures_path = output_path = loss_log_path = None
    if save_mode and dirs is not None:
        figures_path = dirs.figure_path / f"T={cfg.model.T}"
        figures_path.mkdir(parents=True, exist_ok=True)
        output_path = dirs.figure_path / f"outputs_T={cfg.model.T}.txt"
        loss_log_path = dirs.loss_log_path

    stats, stats_eos, total_eos, sequences, _ = evaluation_from_generation(
        model,
        grammar,
        evaluation_dataset=dataset,
        T=cfg.model.T,
        strategy=strategy,
        temperature=cfg.temperature,
        write_steps=save_mode,
        device=device,
        figures_path=figures_path,
        loss_log_path=loss_log_path,
        output_path=output_path,
        save_mode=save_mode,
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


def plot_accuracy_vs_T(all_metrics):
    """
    Line chart of both-rules accuracy vs T on a log x-axis.

    Returns a wandb.Image of the matplotlib figure.
    Also returns two parallel lists (T_values, acc_all, acc_fin) so the caller
    can additionally log a native wandb interactive chart.
    """
    T_values = [m["T"] for m in all_metrics]
    acc_all  = [m["Rule_Accuracy/both_rules_acc"]          for m in all_metrics]
    acc_fin  = [m["Rule_Accuracy_Finished/both_rules_acc"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Lines ──────────────────────────────────────────────────────────────────
    ax.plot(T_values, acc_all, "o-",  color="#4C72B0", linewidth=2, markersize=8,
            label="All sequences", zorder=3)
    ax.plot(T_values, acc_fin, "s--", color="#DD8452", linewidth=2, markersize=8,
            label="Finished (EOS present)", zorder=3)

    # ── Per-point T labels ─────────────────────────────────────────────────────
    # Alternate above/below to avoid overlap when points are close vertically.
    for i, (t, y) in enumerate(zip(T_values, acc_all)):
        offset = 11 if i % 2 == 0 else -16
        ax.annotate(
            f"T={t}", xy=(t, y), xytext=(0, offset),
            textcoords="offset points", ha="center", fontsize=8,
            color="#4C72B0", fontweight="bold",
        )
    for i, (t, y) in enumerate(zip(T_values, acc_fin)):
        offset = -16 if i % 2 == 0 else 11
        ax.annotate(
            f"T={t}", xy=(t, y), xytext=(0, offset),
            textcoords="offset points", ha="center", fontsize=8,
            color="#DD8452", fontweight="bold",
        )

    # ── Axes / grid ────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xticks(T_values)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()                          # suppress unhelpful minor ticks on log axis
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.set_xlabel("T  —  denoising steps  (log scale)", fontsize=11)
    ax.set_ylabel("Both-rules accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Oracle: both-rules accuracy vs. T  (unconditional dataset)", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img, T_values, acc_all, acc_fin


def _wandb_line_chart(T_values, acc_all, acc_fin):
    """
    Build a native W&B interactive line chart (wandb.plot.line_series).

    W&B's line_series uses a shared x-axis list and one y-list per series.
    We pass log10(T) as x so the interactive chart approximates a log scale,
    and include the true T value in every x-tick label.
    """
    log_T = [round(float(np.log10(t)), 4) for t in T_values]
    x_labels = [f"T={t}" for t in T_values]   # shown in the hover tooltip via keys

    return wandb.plot.line_series(
        xs=log_T,
        ys=[acc_all, acc_fin],
        keys=["All sequences", "Finished (EOS present)"],
        title="Oracle: both-rules accuracy vs. T  (log scale)",
        xname="log₁₀(T)",
    )


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
    parser.add_argument("--save", action="store_true", help="Save outputs and figures to disk")
    parser.add_argument(
        "--grammar", type=str, default=None,
        help="Override cfg.data.grammar (e.g. baN, aNbNcN, parentheses_and_brackets).",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    cfg = dict_to_ns(base_config)

    if args.T is not None:
        cfg.model.T = args.T
    if args.grammar is not None:
        cfg.data.grammar = args.grammar

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
        # Ensure model.T is visible in the run config so the sweep dashboard
        # can use it as the x-axis for auto-generated panels.
        # Use a flat underscore key — dotted keys create nested W&B config
        # objects (model.T.value) which the scatter panel cannot resolve.
        wandb.config.update({"model_T": cfg.model.T}, allow_val_change=True)
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
    PROJECT_ROOT = Path(args.config).resolve().parent
    dirs = setup_dirs(PROJECT_ROOT, cfg, args.config, save_mode=args.save)

    grammar = get_grammar(cfg.data.grammar, cfg.data.l)
    grammar.generate_seq()
    schedule = get_schedule(cfg)

    if args.all_T:
        # Multi-T mode: log each T as a step so W&B draws line charts
        all_metrics = []
        for t_val in args.all_T:
            cfg.model.T = t_val
            print(f"\nRunning T={t_val}…")
            m = eval_one_T(cfg, device, grammar, schedule, dirs=dirs, save_mode=args.save)
            all_metrics.append(m)
            flat = {k: v for k, v in m.items() if k != "T"}
            wandb.log({"T": t_val, **flat}, step=t_val)
            print(f"  both_rules_acc={m['Rule_Accuracy/both_rules_acc']:.4f}  "
                  f"finished_pct={m['finished_pct']:.4f}")

        img, T_values, acc_all, acc_fin = plot_accuracy_vs_T(all_metrics)
        wandb.log({
            # Static matplotlib image — clean, annotated, log-scale x-axis
            "accuracy_vs_T/image": img,
            # Native W&B interactive line chart — hover tooltips, zoom, export
            "accuracy_vs_T/interactive": _wandb_line_chart(T_values, acc_all, acc_fin),
        })
        plt.close("all")

    else:
        # Single-T mode (sweep agent or --T override)
        print(f"Running oracle evaluation: T={cfg.model.T}, device={device}")
        m = eval_one_T(cfg, device, grammar, schedule, dirs=dirs, save_mode=args.save)
        wandb.log(m)
        # Write summary explicitly so the sweep overview panel can plot these
        # metrics against model_T across runs. Keys must exactly match what the
        # sweep yaml declares under `metric.name` (no eval_ prefix).
        wandb.summary.update({
            "model_T":                                       cfg.model.T,
            "both_rules_acc":                                m["Rule_Accuracy/both_rules_acc"],
            "both_rules_acc_finished":                       m["Rule_Accuracy_Finished/both_rules_acc"],
            "finished_pct":                                  m["finished_pct"],
            "Rule_Accuracy/both_rules_acc":                  m["Rule_Accuracy/both_rules_acc"],
            "Rule_Accuracy_Finished/both_rules_acc":         m["Rule_Accuracy_Finished/both_rules_acc"],
        })
        print(f"\n=== Oracle T={cfg.model.T} Results ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.finish()


if __name__ == "__main__":
    main()
