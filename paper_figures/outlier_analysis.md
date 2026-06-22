# Outlier & anomaly analysis

Source: `results/combined_6_grammar.csv` (oracle sweep, 1468 rows). All figures regenerated from this file by `make_paper_figures.py`.

## 0. Two length regimes (not an anomaly — document it)
- Single-string grammars (baᴺ, bbaᴺ, aᴺbᴺ, aᴺbᴺcᴺ) swept at **L=128** (max T≈130).
- Dyck-2 grammars (nested / flat) swept at **L=32** (max T≈34).
- Consequence: in Fig 1 the Dyck panels only span the low-NFE region; the compute axis must be read per-panel (L is annotated).

## 1. baᴺ — the parity anomaly (the headline outlier)
- **Degeneracy is collapse, not a T value.** A single repeated valid string can appear at T=1, 2 or 4 (e.g. bbaᴺ greedy at T=2 has uniqueness 0.056). So instead of excluding low-T rows, we compare samplers **at maximum compute**, where greedy is fully iterative and diverse — a degeneracy-free comparison (AR/EB excluded; they reach 1.0 trivially).
  - Under **uniform** (max compute), categorical beats greedy for: ['baN', 'parentheses_and_brackets', 'not_nested_parentheses_and_brackets'] (baᴺ 0.884 vs 0.788).
  - Under **both** uniform and Gaussian, categorical wins for: ['baN', 'not_nested_parentheses_and_brackets'].
- **baᴺ is the strongest, not the unique, case**: positive under both schedules with the widest margin; greedy wins all three counting grammars (aᴺbᴺ, aᴺbᴺcᴺ, bbaᴺ); Dyck is mixed (nested flips to greedy under Gaussian; flat is a +0.002 tie). An earlier draft over-claimed 'baᴺ only' by counting greedy's degenerate one-shot collapse as a win.
- Uniform categorical at T=1 sits at **0.502** ≈ ½, the chance rate of a single parity bit; uniform greedy at T≤8 is **0.000** (deterministic wrong parity).
- **Why:** baᴺ = 'starts with b, even number of a's'. The even-count rule is a global parity constraint that couples positions. The oracle's *per-position marginal* for an interior slot is ~½/½ (either symbol can be valid depending on the rest), so:
  - **greedy** (argmax per position, all at once at low T) commits to one fixed pattern whose parity is wrong with near-certainty → ~0 accuracy until T is large enough that tokens are committed sequentially and the marginal re-conditions.
  - **categorical** samples each undecided position ~½/½, so the final parity is a fair coin → ~0.5 even at T=1, then climbs as conditioning kicks in.
- This is the cleanest evidence in the paper that the failure is a **sampling** phenomenon (marginal vs joint), not a denoiser-capacity one — the denoiser is exact.

## 2. bbaᴺ / Dyck-flat — greedy reaches 1.0 at T=1 (lucky collapse, not skill)
- bbaᴺ: greedy T=1 accuracy **1.000**, uniqueness **0.01** → every sample is the SAME modal string.
- Dyck-2 flat: greedy T=1 accuracy **1.000**, uniqueness **0.002** → every sample is the SAME modal string.
- **Why:** at T=1 greedy argmaxes the fully-masked marginal in one shot. If the modal per-position configuration happens to be a valid string for that grammar, accuracy is trivially 1.0 — but it is a single point mass (zero diversity). Contrast aᴺbᴺ/baᴺ where the modal one-shot string is invalid → T=1 greedy = 0. So 'accuracy at T=1' must always be read together with diversity; report both or it is misleading.

## 3. Categorical's accuracy is real but expensive
- High-accuracy categorical points sit at large NFE (median 125, up to 250 steps), whereas greedy reaches comparable or higher accuracy at a fraction of the compute. This is the core compute–quality asymmetry behind the trade-off figures.

## 4. EB-sampler categorical underperforms on the counting grammars
- aᴺbᴺ EB categorical best = **0.630** vs EB greedy = 1.000. The adaptive entropy-bounded stopping halts while the count is still uncertain under stochastic sampling; greedy is unaffected because it never injects sampling noise. Flag in text; do not present EB as uniformly Pareto-dominant.

## 5. Diversity reliability gate
- **297/1468** rows are flagged `n_correct_too_low` (too few valid samples to estimate diversity); their diversity is set to NaN and omitted from Fig 2/3. Most are low-compute or collapsed-greedy cells. This is why several greedy lines in Fig 2 are short — by construction, not by omission.

## 6. Is more compute always better? (monotonicity check on the envelope)
- 10 envelope line(s) peak at intermediate compute (worth a sentence):
  - baN / gaussian / greedy: acc@max-compute=0.872 vs peak=0.938
  - baN / gaussian / categorical: acc@max-compute=0.916 vs peak=0.972
  - bbaN / uniform / greedy: acc@max-compute=0.996 vs peak=1.000
  - bbaN / gaussian / greedy: acc@max-compute=0.884 vs peak=1.000
  - aNbN / gaussian / categorical: acc@max-compute=0.956 vs peak=0.962
  - aNbNcN / gaussian / categorical: acc@max-compute=0.942 vs peak=0.952
  - parentheses_and_brackets / gaussian / greedy: acc@max-compute=0.910 vs peak=0.988
  - not_nested_parentheses_and_brackets / uniform / greedy: acc@max-compute=0.884 vs peak=1.000
  - not_nested_parentheses_and_brackets / gaussian / greedy: acc@max-compute=0.908 vs peak=1.000
  - not_nested_parentheses_and_brackets / gaussian / categorical: acc@max-compute=0.918 vs peak=0.942
