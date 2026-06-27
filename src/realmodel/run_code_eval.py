"""Generate completions across the decoder/sampler/NFE grid for EvalPlus.

Run on the GPU box. Writes one ``samples.jsonl`` per config under
``<out_dir>/<benchmark>/<config_tag>/``, ready for ``evalplus.sanitize`` +
``evalplus.evaluate``.

Example (pilot):
    python -m realmodel.run_code_eval --benchmark humaneval \
        --decoders uniform gaussian --samplers greedy \
        --nfes 16 32 --sigmas 8 32 --limit 20 --out-dir results_realmodel

Then evaluate (see aggregate_results.py / README).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product

import torch

from realmodel.coda_denoiser import CodaDenoiser
from realmodel.decode import DecodeConfig, generate


# ── instruction templates ────────────────────────────────────────────────────

def humaneval_instruction(problem: dict) -> str:
    return ("Complete the following Python function. Return only the complete "
            "function in a markdown code block.\n\n" + problem["prompt"])


def mbpp_instruction(problem: dict) -> str:
    return problem["prompt"]  # evalplus mbpp-plus prompt is already self-contained


def load_problems(benchmark: str):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    if benchmark == "humaneval":
        return get_human_eval_plus(), humaneval_instruction
    if benchmark == "mbpp":
        return get_mbpp_plus(), mbpp_instruction
    raise ValueError(benchmark)


# ── grid expansion ───────────────────────────────────────────────────────────

def build_grid(args) -> list[DecodeConfig]:
    cfgs: list[DecodeConfig] = []
    for decoder, sampler in product(args.decoders, args.samplers):
        if decoder == "uniform":
            for nfe in args.nfes:
                cfgs.append(DecodeConfig("uniform", sampler, nfe=nfe, **_common(args, sampler)))
        elif decoder == "gaussian":
            for nfe, sigma in product(args.nfes, args.sigmas):
                cfgs.append(DecodeConfig("gaussian", sampler, nfe=nfe, sigma=sigma,
                                         **_common(args, sampler)))
        elif decoder == "eb":
            for gamma in args.gammas:
                cfgs.append(DecodeConfig("eb", sampler, gamma=gamma, **_common(args, sampler)))
        elif decoder == "ar":
            cfgs.append(DecodeConfig("ar", sampler, **_common(args, sampler)))
        else:
            raise ValueError(decoder)
    return cfgs


def _common(args, sampler: str) -> dict:
    return dict(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples if sampler == "categorical" else 1,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--model", default="Salesforce/CoDA-v0-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-dir", default="results_realmodel")
    ap.add_argument("--decoders", nargs="+", default=["uniform", "gaussian", "eb"])
    ap.add_argument("--samplers", nargs="+", default=["greedy", "categorical"])
    ap.add_argument("--nfes", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[2, 8, 32, 128])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5, 2, 5])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=5,
                    help="i.i.d. samples per problem for categorical (greedy forced to 1)")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (0=all)")
    args = ap.parse_args()

    den = CodaDenoiser.load(args.model, device=args.device, dtype=args.dtype)
    problems, instr_fn = load_problems(args.benchmark)
    task_ids = list(problems.keys())
    if args.limit:
        task_ids = task_ids[: args.limit]

    cfgs = build_grid(args)
    print(f"[run] benchmark={args.benchmark} problems={len(task_ids)} configs={len(cfgs)}")

    for ci, cfg in enumerate(cfgs):
        out_dir = os.path.join(args.out_dir, args.benchmark, cfg.tag())
        os.makedirs(out_dir, exist_ok=True)
        samples_path = os.path.join(out_dir, "samples.jsonl")
        nfe_log: list[int] = []
        t0 = time.time()
        with open(samples_path, "w") as f:
            for task_id in task_ids:
                res = generate(den, instr_fn(problems[task_id]), cfg)
                nfe_log.append(res.realised_nfe)
                for sol in res.completions:
                    f.write(json.dumps({"task_id": task_id, "solution": sol}) + "\n")
        mean_nfe = sum(nfe_log) / max(1, len(nfe_log))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"config": cfg.__dict__, "mean_nfe": mean_nfe,
                       "n_problems": len(task_ids)}, f, indent=2)
        print(f"[{ci+1}/{len(cfgs)}] {cfg.tag():40s} mean_nfe={mean_nfe:6.1f} "
              f"({time.time()-t0:5.1f}s) -> {samples_path}")


if __name__ == "__main__":
    main()
