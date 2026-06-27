# Handoff — code-benchmark GPU session

This code was written entirely on a Mac and **has never run on a GPU**. The
run order in `README.md` is correct; this file adds the "stop/debug" discipline
and lists the specific interface guesses that are likely wrong.

Work through the gates in order. A later gate is meaningless if an earlier one
failed.

---

## Setup

```bash
git checkout code-benchmark && git pull
python -m venv .venv && source .venv/bin/activate
pip install "torch" "transformers>=4.44" accelerate "evalplus[vllm]"
export PYTHONPATH=src
nvidia-smi   # confirm GPU + free VRAM (model needs ~4GB)
```

---

## Gate 1 — Interface verification (highest risk)

```bash
python -m realmodel.sanity_check
```

The harness guesses CoDA's API in two places that are very likely wrong:

**Guess 1 — `coda_denoiser.py::logits()`**

```python
out = self.model(input_ids=canvas)
return out.logits          # shape (B, L, V)
```

Assumes `model(input_ids=<mask-filled canvas>)` returns per-position full-vocab
logits. CoDA may instead expect `attention_mask`, or may use a different key.
Read the traceback / `model.forward` signature to find the real call.

**Guess 2 — `sanity_check.py` native-generate call**

```python
model.generate(ids, max_tokens=256, diffusion_steps=64, temperature=0.0)
```

`max_tokens` / `diffusion_steps` / `temperature` are guessed kwarg names.
If it errors, check CoDA's `generation_utils.py` or model card for the real
kwargs (likely `max_new_tokens`, `num_diffusion_steps` or similar).

**Token ids**

CoDA sets `generation_config.mask_token_id`. Sanity-check prints `mask_id`,
`eos_id`, `pad_id`, `vocab` — confirm they look sane (mask_id should be a
single high-index token, not 0 or None).

**Success criterion:** both the native `generate` call and our `uniform/greedy`
decode loop emit a plausible Python function on the test prompt.

If you change `coda_denoiser.py`, keep edits minimal and add a comment
explaining what CoDA actually does (so the diff is self-documenting).

---

## Gate 2 — Reproduce the published number (trust anchor)

EB/confidence is CoDA's native sampler, so `eb/greedy/γ=0.5` should land
within a few points of CoDA's published HumanEval pass@1 (~54 %).

```bash
python -m realmodel.run_code_eval --benchmark humaneval \
    --decoders eb --samplers greedy --gammas 0.5 --out-dir results_realmodel

python -m realmodel.aggregate_results --benchmark humaneval \
    --out-dir results_realmodel --csv results_realmodel/humaneval_sanity.csv
```

**Common failure: `pass@1=None` in the CSV.** This means `aggregate_results.py`
didn't parse evalplus output. The regex looks for `pass@1:` — if evalplus changed
its stdout format, fix the parser against the real output before continuing.

**Stop if the number is far from ~54 %.** Every schedule comparison downstream
is meaningless until this gate passes.

---

## Gate 3 — Pilot (cheap direction check, ~40 problems)

```bash
python -m realmodel.run_code_eval --benchmark humaneval \
    --decoders uniform gaussian --samplers greedy \
    --nfes 16 32 --sigmas 8 32 --limit 40 --out-dir results_realmodel

python -m realmodel.aggregate_results --benchmark humaneval \
    --out-dir results_realmodel --csv results_realmodel/humaneval_pilot.csv
```

Sanity question: does Gaussian ≥ uniform at low NFE (the paper's claim)?
If the direction is reversed, understand why before scaling up.

---

## Gate 4 — Full grid, both benchmarks

Only after gates 1–3 look good.

```bash
for B in humaneval mbpp; do
  python -m realmodel.run_code_eval --benchmark $B \
      --decoders uniform gaussian eb ar --samplers greedy categorical \
      --nfes 8 16 32 64 --sigmas 2 8 32 128 --gammas 0.1 0.5 2 5 \
      --num-samples 5 --out-dir results_realmodel
  python -m realmodel.aggregate_results --benchmark $B \
      --out-dir results_realmodel --csv results_realmodel/${B}_passk.csv
done
```

**Multi-GPU:** shard by benchmark or decoder across `CUDA_VISIBLE_DEVICES=k`
parallel shells. Each process loads its own 1.7B copy; it fits on a single GPU.
Watch wall-clock for the first config and extrapolate before committing to the
full grid.

---

## Known gotchas

| Issue | Detail |
|---|---|
| σ units | σ is in canvas-position units (0..`max_new_tokens`=256), not the L=128 grammar grid. Re-tune σ if `max_new_tokens` changes. |
| NFE accounting | EB/AR are adaptive. `meta.json` logs mean realised NFE per config. For the headline comparison, filter EB points that land in the same NFE band as uniform/Gaussian. |
| Instruct extraction | Completions are raw chat text. `evalplus.sanitize` extracts the runnable function. If it yields empty solutions, the markdown-codeblock stripping may need adjustment. |
| Decode loop | `decode.py` has per-row Python loops (no CUDA kernel). Fine for correctness. Only optimize if throughput is the bottleneck — do not touch schedule math. |

---

## Definition of done

1. `results_realmodel/humaneval_passk.csv` and `results_realmodel/mbpp_passk.csv`
   with real (non-None) pass@1 / pass@1_plus across the grid.
2. The EB sanity config (Gate 2) reproduces ~54 % on HumanEval.
3. Short written summary: does Gaussian/structured ≥ uniform at low NFE on a real
   model? Include the numbers.
4. Commit results + any interface fixes to `code-benchmark` and push. Do not
   commit the multi-GB model cache (already in `.gitignore`).

Report back after each gate rather than running silently to the end.
