# Research-pipeline audit — pre-final-run

Audit of the oracle sampling-dynamics pipeline that produces `results/combined_6_grammar.csv`
and the paper. Triggered by the `does_satisfy_format` bug; goal is to stop similar bugs
leaking into the final paper. Five subsystems audited in depth (oracle, rules/format/dataset,
schedules, unmasking loop, sweep-driver + aggregation + diversity). Findings below are
labelled **Confirmed** (reproduced), **Likely**, or **Suspected**.

**Verdict:** the scientific foundation is sound — the analytic oracle is correct under
exhaustive brute-force, the NFE/compute axis is honest, rule definitions are verbatim-faithful,
and aggregation is correct. Two issues had already leaked into the paper draft (both now fixed)
and one eval bug was fixed in code. The rest are design choices / latent footguns to confirm
before the final run.

---

## A. Already in the paper draft — FIXED

| # | Sev | What | Fix |
|---|-----|------|-----|
| **P1** | High | **"Uniform is the large-σ limit of the Gaussian schedule" is mathematically false.** With `W=L+4σ`, as σ→∞ the schedule becomes position-independent but its time profile tends to `Φ(4t−2)`, *not* the uniform `p_mask=t`. Verified numerically: σ=1e5 gives p_mask = [0.023, 0.159, 0.500, 0.841, 0.977] = Φ(4t−2) exactly. | `paper.typ` §7.1, fig9 caption, conclusions: reworded to "position-independent, uniform-*like* limit `Φ(4t−2)`"; uniform kept as a separate baseline. |
| **P2** | High | **Accuracy error bands used the noisy 5-rep `std`, not the Wilson CI the driver computed for exactly this reason.** `std_both_rules==0` on 96/618 stochastic categorical cells → false zero-width bars (e.g. bbaᴺ T=1: std=0 but CI=[0, 0.0076]). The `ci_low/high_both_rules` columns exist on all 1236 rows but `make_paper_figures.py` never referenced them. | `make_paper_figures.py` `load()` + fig1 now use `ci_low/high_both_rules` (fallback to mean±std); fig1 caption updated to "95% Wilson CI". |

## B. Pipeline code — FIXED

| # | Sev | What | Fix |
|---|-----|------|-----|
| **B1** | Medium | **`re_grammar.evaluate()` ran rule checks on the full padded sequence**, so content after a valid EOS leaked into `mean_rule1`/`mean_rule2` (diverging from `anbn.evaluate`, which truncates). `both_rules`/`grammatical` was already safe (format rejects trailing content). | `re_grammar.evaluate()` now runs r1/r2 on the content up to the first EOS, while format still checks the full sequence. Verified `grammatical` unchanged; per-rule counts no longer leak. 38 oracle tests pass. |

## C. Pipeline — resolutions

| # | Sev | What | Resolution |
|---|-----|------|------------|
| **C1** | Med | **Temperature was a silent no-op on the oracle path** (`probs = raw/temperature`, no softmax; `torch.multinomial` renormalizes, so τ cancels). Harmless as configured (τ=1.0). | **FIXED** (`noise_schedule_unmask.py` `_content_probs`): oracle temperature now applied in probability space as `p^(1/T)/Σ`, which is the identity at T=1 (so no current/final result changes — all configs use τ=1.0) and **preserves exact zeros** so invalid tokens are never sampled. Verified numerically. |
| **C4** | Low | Audit flagged **AR-categorical as "effectively deterministic"** (all `std_both_rules=0`) → suspected wasted compute / mislabeled condition. | **RESOLVED — not a defect.** On verification, categorical-AR is genuinely stochastic: uniqueness 1.0/0.928/0.128 vs greedy-AR's 0.010 — it is the *diverse-AR reference* §4 relies on. `std_both_rules=0` only reflects AR being a perfect generator (accuracy always 1.0); the 5 reps are needed for the diversity error bars. `is_deterministic` is correct. **No change made** (setting n_reps=1 would have destroyed the diversity baseline). |
| **C3** | Med | **Heterogeneous grid:** non-Dyck at **L=128, T≤1024**; Dyck at **L=32, T≤512**. Cross-grammar matched-compute comparisons are confounded. | **DOCUMENTED** (per user: do NOT re-run Dyck). `paper.typ` §1 now states both the L and T-ceiling difference and that all figures are read per-panel with no cross-grammar matched-compute claim. |
| **C2** | Low-Med | **Mop-up fill** fills leftover MASKs in one independent pass when the loop didn't finish; can yield invalid tail sequences. Correctly counted as +1 NFE, respects the sampler. | **Not a defect** — this is honest sampling behavior (a sampler that paints itself into a dead-end correctly scores non-grammatical; format rejects any leftover MASK). Left as-is. Optional: add a mop-up counter if you want to report its frequency. |
| **C5** | Low | **`n_samples=100`/rep** (script help advertises 500); 196/1236 rows trip the `n_correct<5` diversity gate. | **Left to you** — a run-scale config choice, not a code bug. Raising it tightens CIs and ungates more diversity cells. Not changed (you did not ask to change run scale). |
| **C6** | Low | **Per-rep seeds shared across all cells of a grammar.** | **Left intentionally** — this *helps* the paired greedy-vs-categorical contrasts (shared noise cancels), which the figures use. Not a defect. |

## D. Operational checklist for the final run

- **`python-Levenshtein`**: already listed in `requirements.txt` (line 9); it was merely not
  installed in the local `venv` at audit time. **RESOLVED** — installed and verified importable
  in `venv` (`Levenshtein`, `pandas`, `seaborn` all import). No requirements change needed.
- **Driver = `src/oracle/eval_oracle_T_sweep.py`** (confirmed by exact CSV column match). The
  final CSV is a merge of two runs (4 non-Dyck @ L=128, 2 Dyck @ L=32). Resume logic appends
  keyed on the full cell tuple; current CSV has **0 duplicate cells**.
- **A cell that raises is silently dropped** (no row written), not turned into a garbage row.
  After the run, assert `row_count == expected_grid_size` to catch silent drops.
- **σ grid** in the actual driver is `[1,2,5,10,20,50,100,256]` (matches the CSV). The
  unrelated `eval_oracle_categorical_sweep.py` has a docstring/code σ-grid mismatch — it is
  **not** the driver, so it does not affect the final CSV; ignore or fix separately.

## E. Verified CORRECT (reassurance — coverage)

- **Oracle denoiser**: exhaustive L=8 brute-force across all six grammars, 0 value/None
  mismatches; conditions on revealed context (not just the marginal); rows normalize to 1;
  never assigns mass to MASK; dead-ends return `(None, msg)` with no NaN/inf; deterministic
  (big-int counting). 57 oracle tests pass. *This is the paper's foundation and it holds.*
- **Gaussian schedule math**: `dp_mask = φ·W` carries the full `W/σ` chain-rule factor
  (finite-difference-verified <0.1% error across σ∈{1,5,20,256}); `margin=2σ`, `width=L+4σ`
  matches the source note. Categorical `p_mask=t`, `dp=1` correct.
- **NFE / compute axis**: `n_steps_mean` == oracle forward passes, no off-by-one; realised
  steps ≤ requested T (early termination); EB's T correctly ignored; greedy T=1 = one-shot
  argmax of the fully-masked marginal (mode-collapse claim holds). Figures plot realised
  `n_steps_mean`, not requested T (verified on early-termination rows).
- **Rules**: bit-identical to `re_data.grammar_rules` over 20k random seqs/grammar (0
  mismatches); all generated targets round-trip to `grammatical=1`; Dyck nested-vs-flat
  distinction correct (`([)]` accepted by flat, rejected by nested).
- **Format check** (the original bug): now rejects empty `SOS,EOS`, any MASK, mis-ordered or
  multi SOS/EOS, PAD-in-content, and content-after-EOS; round-trip-safe on all generators.
- **Aggregation**: `n_correct == mean_both_rules × n_eval_total` exactly on all 1236 rows;
  `both_rules = r1 & r2 & fmt`; `n_correct_too_low` gate (n<5) has 0 mismatches (196 flagged);
  NaN written as empty → figures gate diversity correctly; `sigma`/`eb_gamma` populated only
  for the right decoder; 0 duplicate cells.

## F. Dead code (no impact; left per surgical-change policy)

- `src/schedules/tanh_schedule.py` is a byte-for-byte duplicate of `GaussianSchedule`
  (misnamed), not exported, unused.
- `grammar_oracles.py` vestigial `_dyck_forward`/`_dyck_backward`/`_initial_state` (live code
  uses the `*_impl` variants).
