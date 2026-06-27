# Real-model stress-test (CoDA-1.7B)

Swaps the paper's grammar **oracle** for a real pre-trained masked-diffusion code
model (`Salesforce/CoDA-v0-Instruct`, ~1.7B) and evaluates the paper's unmasking
schedules / samplers on **HumanEval(+)** and **MBPP(+)** via **EvalPlus**.

The schedule + sampler math (`schedules.py`) is ported 1:1 from `src/schedules/`
but decoupled from the grammar token conventions, so the same logic drives a
real BPE-vocab model. One unified loop (`decode.py`) runs **every** decoder
(uniform / Gaussian / EB / AR) so the comparison is fair: 1 forward = 1 NFE.

## Files
- `coda_denoiser.py` — loads CoDA, resolves mask/eos ids, builds chat prompts,
  exposes `logits(canvas)` (the `oracle=False` drop-in: returns full-vocab logits).
- `schedules.py` — ported Gaussian/uniform `p_mask`, schedule/EB/AR position
  selectors, greedy/categorical samplers.
- `decode.py` — unified canvas decoding loop + `DecodeConfig`.
- `run_code_eval.py` — generate completions across the config grid → `samples.jsonl`.
- `aggregate_results.py` — `evalplus.sanitize` + `evalplus.evaluate` → pass@1 CSV.
- `sanity_check.py` — verification step 1 (token ids + native-vs-our-loop).

## Environment (GPU box — NOT the local mac)
```bash
pip install "torch" "transformers>=4.44" accelerate "evalplus[vllm]" || pip install evalplus
# CoDA uses trust_remote_code; first load downloads ~3.4GB weights.
export PYTHONPATH=src        # so `python -m realmodel.*` resolves the package
```

## Run order

### 1. Sanity / interface check (do this first)
```bash
python -m realmodel.sanity_check
```
Confirm the printed `mask_id`/`eos_id`/`vocab` look right and that both the
native `generate` and our `uniform/greedy` loop emit a plausible function. If the
native call fails, read the printed error for the real `generate` signature and
adjust `coda_denoiser.py` / `sanity_check.py`.

### 2. Reproduce CoDA's published HumanEval (~54%) with the default-style sampler
```bash
python -m realmodel.run_code_eval --benchmark humaneval \
    --decoders eb --samplers greedy --gammas 0.5 --out-dir results_realmodel
python -m realmodel.aggregate_results --benchmark humaneval \
    --out-dir results_realmodel --csv results_realmodel/humaneval_sanity.csv
```
EB/confidence ≈ CoDA's native sampler; pass@1 should land within a few points of
~54%. **Do not trust schedule comparisons until this matches.**

### 3. Pilot grid (cheap signal check)
```bash
python -m realmodel.run_code_eval --benchmark humaneval \
    --decoders uniform gaussian --samplers greedy \
    --nfes 16 32 --sigmas 8 32 --limit 40 --out-dir results_realmodel
python -m realmodel.aggregate_results --benchmark humaneval \
    --out-dir results_realmodel --csv results_realmodel/humaneval_pilot.csv
```
Check the direction: does Gaussian ≥ uniform at low NFE?

### 4. Full grid (low–mid NFE), both benchmarks
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

## Notes
- **NFE accounting:** every forward pass increments NFE; `meta.json` records the
  mean realised NFE per config (EB/AR are adaptive). Keep EB points that land in
  the low–mid band for the headline comparison.
- **σ units:** σ is in canvas-position units (0..`max_new_tokens`), *not* the
  L=128 grammar grid. Re-tune if you change `max_new_tokens`.
- **Multi-GPU:** the simplest scaling is to shard `--decoders`/benchmarks across
  GPUs with `CUDA_VISIBLE_DEVICES=k` in parallel shells; each process loads its
  own copy of the 1.7B model (fits comfortably).
- **Instruct extraction:** completions are raw chat output; `evalplus.sanitize`
  extracts the runnable function before `evaluate`.
