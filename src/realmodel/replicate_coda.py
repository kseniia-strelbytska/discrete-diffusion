"""Replicate CoDA's published HumanEval result as a trust anchor.

Mirrors the official eval script (CoDA/evaluation/lm_eval/eval_mbpp_humaneval.sh):
  alg="entropy", temperature=0.1, top_p=0.9, steps=768, max_new_tokens=768
  prompt = chat-template(user_msg) + gen_prefix (assistant prefix with function signature)

Run:
    python -m realmodel.replicate_coda [--limit N] [--out-dir results_realmodel]

If pass@1 lands in [0.50, 0.57] the trust anchor holds and we can proceed to
swap in our custom schedules.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import re

import torch

from realmodel.coda_denoiser import CodaDenoiser

# ── Official CoDA prompt format (from humaneval_instruct.yaml) ────────────────

def make_prompt(tok, problem: dict) -> torch.Tensor:
    """Build prompt_ids matching CoDA's official humaneval_instruct evaluation."""
    user_msg = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```{problem['prompt']}"
    )
    gen_prefix = f"Here is the completed function:\n```python\n{problem['prompt']}\n"

    chat_str = tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    full_str = chat_str + gen_prefix
    ids = tok.encode(full_str, return_tensors="pt")[0]
    return ids


def postprocess(response: str, prompt: str, entry_point: str) -> str:
    """Mirror build_predictions_instruct + sanitize from utils.py."""
    # extract code from the markdown block the model generates
    body = response.split('```python\n', 1)[-1].split('```')[0]
    code = prompt + "\n" + body
    return sanitize(code, entry_point)


def sanitize(code: str, entry_point: str) -> str:
    """Minimal sanitize: keep only the target function (mirrors sanitize_utils.py logic)."""
    # evalplus.sanitize does the heavy lifting at eval time; here just strip trailing noise
    lines = code.split("\n")
    out, in_fn, indent0 = [], False, None
    for line in lines:
        if not in_fn:
            out.append(line)
            if line.startswith(f"def {entry_point}"):
                in_fn = True
                indent0 = len(line) - len(line.lstrip())
        else:
            stripped = line.lstrip()
            if stripped == "":
                out.append(line)
                continue
            cur_indent = len(line) - len(stripped)
            if cur_indent <= indent0 and stripped and not stripped.startswith("#"):
                break
            out.append(line)
    return "\n".join(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Salesforce/CoDA-v0-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--alg", default="entropy")
    ap.add_argument("--limit", type=int, default=0, help="first N problems (0=all)")
    ap.add_argument("--out-dir", default="results_realmodel")
    args = ap.parse_args()

    den = CodaDenoiser.load(args.model, device=args.device, dtype=args.dtype)
    tok = den.tokenizer

    # load problems (avoid src/datasets/ shadowing)
    import sys
    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _had = _src in sys.path
    if _had:
        sys.path.remove(_src)
    try:
        from evalplus.data import get_human_eval_plus
    finally:
        if _had:
            sys.path.insert(0, _src)

    problems = get_human_eval_plus()
    task_ids = list(problems.keys())
    if args.limit:
        task_ids = task_ids[:args.limit]

    tag = f"coda_official_steps{args.steps}_mnt{args.max_new_tokens}"
    out_dir = os.path.join(args.out_dir, "humaneval", tag)
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.jsonl")

    print(f"[replicate_coda] {len(task_ids)} problems, steps={args.steps}, "
          f"max_new_tokens={args.max_new_tokens}, alg={args.alg}, "
          f"temperature={args.temperature}, top_p={args.top_p}")

    with open(samples_path, "w") as f:
        for i, tid in enumerate(task_ids):
            prob = problems[tid]
            prompt_ids = make_prompt(tok, prob).unsqueeze(0).to(den.device)

            out = den.model.diffusion_generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
            )
            # decode only the generated portion
            gen_ids = out[0][prompt_ids.shape[1]:]
            response = tok.decode(gen_ids.tolist(), skip_special_tokens=False)
            response = response.split(tok.eos_token)[0]

            solution = postprocess(response, prob["prompt"], prob["entry_point"])

            f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(task_ids)}] {tid}")
                print(f"    response[:200]: {repr(response[:200])}")
                print(f"    solution[:200]: {repr(solution[:200])}")

    print(f"\n[eval] samples -> {samples_path}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(["evalplus.sanitize", samples_path], env=env)
    sanitized = samples_path.replace(".jsonl", "-sanitized.jsonl")
    sanitized = sanitized if os.path.exists(sanitized) else samples_path
    result = subprocess.run(
        ["evalplus.evaluate", "humaneval", "--samples", sanitized],
        capture_output=True, text=True, env=env,
    )
    print(result.stdout + result.stderr)
    nums = re.findall(r"pass@1:\s*([0-9.]+)", result.stdout + result.stderr)
    if nums:
        base, plus = (float(nums[0]), float(nums[1])) if len(nums) >= 2 else (float(nums[0]), None)
        print(f"\n=== RESULT: pass@1={base:.3f}  pass@1+={plus} ===")
        if 0.50 <= base <= 0.57:
            print("TRUST ANCHOR HOLDS ✓")
        else:
            print(f"WARNING: expected 0.50–0.57, got {base:.3f} — investigate before proceeding")
    else:
        print("Could not parse pass@1 from evalplus output")

    json.dump({"steps": args.steps, "max_new_tokens": args.max_new_tokens,
               "alg": args.alg, "temperature": args.temperature,
               "top_p": args.top_p, "n_problems": len(task_ids)},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
