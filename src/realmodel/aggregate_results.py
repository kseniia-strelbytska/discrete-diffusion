"""Sanitize + evaluate every generated config and collect pass@1 into a CSV.

For each ``<out_dir>/<benchmark>/<config_tag>/samples.jsonl`` this runs
``evalplus.sanitize`` then ``evalplus.evaluate`` and parses the base and plus
pass@1, joining with the ``meta.json`` (mean realised NFE). Output is the
analog of the paper's results CSVs.

    python -m realmodel.aggregate_results --out-dir results_realmodel \
        --benchmark humaneval --csv results_realmodel/humaneval_passk.csv
        
    python -m realmodel.aggregate_results --out-dir results_realmodel_v2 \
        --benchmark humaneval --csv results_realmodel_v2/humaneval_passk.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess

PASS1 = re.compile(r"pass@1:\s*([0-9.]+)")


def run(cmd: list[str]) -> str:
    print("  $", " ".join(cmd))
    # Strip PYTHONPATH so src/datasets/ doesn't shadow the pip `datasets` package
    # in evalplus subprocesses.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        print(f"  [WARN] exit {out.returncode}: {(out.stderr or out.stdout)[:300]}")
    return out.stdout + out.stderr


def evaluate_config(cfg_dir: str, dataset: str) -> dict:
    samples = os.path.join(cfg_dir, "samples.jsonl")
    run(["evalplus.sanitize", samples])
    sanitized = samples.replace(".jsonl", "-sanitized.jsonl")
    sanitized = sanitized if os.path.exists(sanitized) else samples
    log = run(["evalplus.evaluate", dataset, "--samples", sanitized])
    nums = PASS1.findall(log)
    base = float(nums[0]) if len(nums) >= 1 else None
    plus = float(nums[1]) if len(nums) >= 2 else None
    return {"pass@1": base, "pass@1_plus": plus}


def parse_meta(cfg_dir: str) -> dict:
    meta_path = os.path.join(cfg_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    c = meta.get("config", {})
    return {
        "decoder": c.get("decoder"), "sampler": c.get("sampler"),
        "nfe_requested": c.get("nfe"), "sigma": c.get("sigma"),
        "gamma": c.get("gamma"), "temperature": c.get("temperature"),
        "num_samples": c.get("num_samples"),
        "mean_nfe": round(meta.get("mean_nfe", 0), 2),
        "n_problems": meta.get("n_problems"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results_realmodel")
    ap.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    base_dir = os.path.join(args.out_dir, args.benchmark)
    cfg_dirs = sorted(
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    rows = []
    for cfg_dir in cfg_dirs:
        print(f"[eval] {cfg_dir}")
        row = {"config": os.path.basename(cfg_dir), "benchmark": args.benchmark}
        row.update(parse_meta(cfg_dir))
        row.update(evaluate_config(cfg_dir, args.benchmark))
        rows.append(row)
        print(f"       pass@1={row.get('pass@1')} plus={row.get('pass@1_plus')}")

    fields = ["benchmark", "config", "decoder", "sampler", "nfe_requested", "sigma",
              "gamma", "temperature", "num_samples", "mean_nfe", "n_problems",
              "pass@1", "pass@1_plus"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
