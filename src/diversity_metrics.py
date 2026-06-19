"""
diversity_metrics.py
====================
Diversity metrics for formal grammar generation experiments.

All metric logic lives here. Grammar classes are thin delegators via their
diversity_metrics() and diversity_distributions() methods.

The dispatch table GRAMMAR_METRIC_SETS is the single source of truth for
which metric sets apply to each grammar. Any logic that needs to know
"does grammar X get metric Y?" must consult this table.

Reference: arXiv:2511.11018 (DFA coverage metrics).
"""

import math
import random
import warnings
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

try:
    import Levenshtein as _lev_mod
except ImportError:
    raise ImportError(
        "python-Levenshtein is required. "
        "Install with: pip install python-Levenshtein"
    )

# ---------------------------------------------------------------------------
# Dispatch table — single source of truth
# ---------------------------------------------------------------------------

GRAMMAR_METRIC_SETS: Dict[str, List[str]] = {
    'baN':                                 ['universal', 'dfa_coverage'],
    'bbaN':                                ['universal', 'dfa_coverage', 'nm_distribution'],
    'aNbN':                                ['universal', 'n_distribution'],
    'parentheses_and_brackets':            ['universal', 'dyck_structure'],
    'aNbNcN':                              ['universal', 'n_distribution'],
    'not_nested_parentheses_and_brackets': ['universal', 'dyck_structure'],
}


# ---------------------------------------------------------------------------
# Sequence preprocessing
# ---------------------------------------------------------------------------

def _strip(seq: Any, vocab: Dict[str, int]) -> Tuple[int, ...]:
    """
    Strip SOS, EOS, and PAD tokens from a sequence.

    Steps:
      1. Remove leading SOS token if present.
      2. Truncate at first EOS token (keep EOS-1 prefix).
      3. Strip any trailing PAD tokens.

    Args:
        seq: 1-D integer tensor, list, or tuple of token IDs.
        vocab: dict with keys 'sos', 'eos', 'pad' mapping to token IDs.

    Returns:
        Tuple[int, ...] representing the stripped content.
    """
    if hasattr(seq, 'tolist'):
        tokens: List[int] = list(seq.tolist())
    else:
        tokens = list(seq)

    sos = vocab.get('sos', 3)
    eos = vocab.get('eos', 2)
    pad = vocab.get('pad', 4)

    # 1. Remove leading SOS
    if tokens and tokens[0] == sos:
        tokens = tokens[1:]

    # 2. Truncate at first EOS
    try:
        eos_idx = tokens.index(eos)
        tokens = tokens[:eos_idx]
    except ValueError:
        pass

    # 3. Strip trailing PAD
    while tokens and tokens[-1] == pad:
        tokens.pop()

    return tuple(tokens)


# ---------------------------------------------------------------------------
# Universal metrics
# ---------------------------------------------------------------------------

def uniqueness(sequences: List[Tuple[int, ...]]) -> float:
    """
    Fraction of unique sequences in the batch.

    Formula:
        uniqueness = |{s : s in S}| / |S|

    Bound: [1/N, 1] where N = len(sequences).

    Args:
        sequences: List of stripped sequences as tuples.

    Returns:
        Float in [1/N, 1], or NaN if sequences is empty.
    """
    n = len(sequences)
    if n == 0:
        return float('nan')
    return len(set(sequences)) / n


def duplication_rate(sequences: List[Tuple[int, ...]]) -> float:
    """
    Fraction of duplicate sequences: 1 - uniqueness.

    Args:
        sequences: List of stripped sequences as tuples.

    Returns:
        Float in [0, 1 - 1/N], or NaN if sequences is empty.
    """
    u = uniqueness(sequences)
    if math.isnan(u):
        return float('nan')
    return 1.0 - u


def mean_lev_dist_normalized(
    sequences: List[Tuple[int, ...]],
) -> Tuple[float, int]:
    """
    Normalized mean pairwise Levenshtein distance (aggregate form per D3LM).

    Formula:
        mean_lev_dist_normalized = sum_{i<j} lev(s_i, s_j) / sum_{i<j}(|s_i|+|s_j|)

    Uses python-Levenshtein (C-backed) for speed. If n > 200, subsamples
    200 sequences with random.Random(42) (fixed seed for reproducibility).
    The actual count used is returned as lev_n_used.

    Bound: [0, 1].

    Args:
        sequences: List of stripped sequences as tuples.

    Returns:
        (metric_value, n_used). Returns (nan, n) if len(sequences) < 2.
    """
    n = len(sequences)
    if n < 2:
        return float('nan'), n

    if n > 200:
        rng = random.Random(42)
        seqs = rng.sample(sequences, 200)
        n_used = 200
    else:
        seqs = list(sequences)
        n_used = n

    total_dist = 0.0
    total_len = 0.0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            # Convert integer tokens to chars (offset avoids control chars)
            s1 = ''.join(chr(t + 128) for t in seqs[i])
            s2 = ''.join(chr(t + 128) for t in seqs[j])
            total_dist += _lev_mod.distance(s1, s2)
            total_len += len(seqs[i]) + len(seqs[j])

    if total_len == 0:
        return 0.0, n_used
    return total_dist / total_len, n_used


def _ngram_diversity(sequences: List[Tuple[int, ...]], k: int) -> float:
    """
    Compute k-gram diversity.

    Formula:
        n-gram_diversity_k = |union_i {k-grams in s_i}| / sum_i |{k-grams in s_i}|

    Sequences shorter than k are skipped (they contribute nothing to numerator
    or denominator).

    Args:
        sequences: List of stripped sequences as tuples.
        k: n-gram size (2 for bigrams, 3 for trigrams).

    Returns:
        Float in [0, 1], or NaN if no sequence has length >= k.
    """
    all_ngrams: Set[Tuple[int, ...]] = set()
    total_count = 0

    for seq in sequences:
        if len(seq) < k:
            continue
        ngrams = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        all_ngrams.update(ngrams)
        total_count += len(ngrams)

    if total_count == 0:
        return float('nan')
    return len(all_ngrams) / total_count


def bigram_diversity(sequences: List[Tuple[int, ...]]) -> float:
    """
    Bigram diversity: |union of bigrams| / sum of bigram counts.

    Sequences of length < 2 are skipped.

    Returns:
        Float in [0, 1], or NaN if no sequence has length >= 2.
    """
    return _ngram_diversity(sequences, 2)


def trigram_diversity(sequences: List[Tuple[int, ...]]) -> float:
    """
    Trigram diversity: |union of trigrams| / sum of trigram counts.

    Sequences of length < 3 are skipped.

    Returns:
        Float in [0, 1], or NaN if no sequence has length >= 3.
    """
    return _ngram_diversity(sequences, 3)


def _compute_universal(stripped: List[Tuple[int, ...]]) -> Dict[str, Any]:
    """
    Compute all universal metrics for pre-stripped sequences.

    Edge-case rules:
      n_correct == 0  : all NaN; n_correct_too_low = True.
      n_correct == 1  : uniqueness = 1.0, lev = NaN, n-grams computed;
                        n_correct_too_low = True.
      2 <= n_correct < 5: all metrics NaN; n_correct_too_low = True.
      n_correct >= 5  : all computed; n_correct_too_low = False.

    Returns:
        Dict with n_correct, n_correct_too_low, uniqueness, duplication_rate,
        mean_lev_dist_normalized, lev_n_used, bigram_diversity, trigram_diversity.
    """
    n = len(stripped)
    nan = float('nan')
    out: Dict[str, Any] = {'n_correct': n}

    if n == 0:
        out.update({
            'n_correct_too_low': True,
            'uniqueness': nan,
            'duplication_rate': nan,
            'mean_lev_dist_normalized': nan,
            'lev_n_used': 0,
            'bigram_diversity': nan,
            'trigram_diversity': nan,
        })
        return out

    if n == 1:
        # Special case: uniqueness and n-grams are well-defined; lev requires >=2
        out.update({
            'n_correct_too_low': True,
            'uniqueness': 1.0,
            'duplication_rate': 0.0,
            'mean_lev_dist_normalized': nan,
            'lev_n_used': 1,
            'bigram_diversity': bigram_diversity(stripped),
            'trigram_diversity': trigram_diversity(stripped),
        })
        return out

    if n < 5:
        out.update({
            'n_correct_too_low': True,
            'uniqueness': nan,
            'duplication_rate': nan,
            'mean_lev_dist_normalized': nan,
            'lev_n_used': n,
            'bigram_diversity': nan,
            'trigram_diversity': nan,
        })
        return out

    # n >= 5: compute everything
    lev_val, lev_n = mean_lev_dist_normalized(stripped)
    out.update({
        'n_correct_too_low': False,
        'uniqueness': uniqueness(stripped),
        'duplication_rate': duplication_rate(stripped),
        'mean_lev_dist_normalized': lev_val,
        'lev_n_used': lev_n,
        'bigram_diversity': bigram_diversity(stripped),
        'trigram_diversity': trigram_diversity(stripped),
    })
    return out


# ---------------------------------------------------------------------------
# DFA state and transition coverage (L1=baN, L2=bbaN)
# Reference: arXiv:2511.11018
# ---------------------------------------------------------------------------

# Module-level cache: grammar_name -> DFA dict. Build once, reuse across calls.
_DFA_CACHE: Dict[str, Dict] = {}

# DFA dict format:
#   states      : list of state IDs
#   alphabet    : list of token IDs (content tokens only; SOS/EOS/PAD stripped)
#   initial     : initial state ID
#   accepting   : frozenset of accepting state IDs
#   transitions : dict {(state_id, token_id): next_state_id}


def _build_ban_dfa() -> Dict:
    """
    Hand-constructed minimal DFA for baN: { B A^{2k} | k >= 0 }.

    Alphabet: A=0, B=1
    States:
      0 = q_start  (initial; rejects on epsilon)
      1 = q_even   (seen B, even # of A's; ACCEPTING)
      2 = q_odd    (seen B, odd # of A's)
      3 = q_dead   (sink)

    Transitions:
      q_start --B--> q_even     (first token must be B)
      q_start --A--> q_dead
      q_even  --A--> q_odd
      q_even  --B--> q_dead     (no second B)
      q_odd   --A--> q_even     (back to even)
      q_odd   --B--> q_dead
      q_dead  --A--> q_dead
      q_dead  --B--> q_dead
    """
    return {
        'states': [0, 1, 2, 3],
        'alphabet': [0, 1],
        'initial': 0,
        'accepting': frozenset([1]),
        'transitions': {
            (0, 0): 3,  # q_start --A--> dead
            (0, 1): 1,  # q_start --B--> q_even
            (1, 0): 2,  # q_even  --A--> q_odd
            (1, 1): 3,  # q_even  --B--> dead
            (2, 0): 1,  # q_odd   --A--> q_even
            (2, 1): 3,  # q_odd   --B--> dead
            (3, 0): 3,  # dead    --A--> dead
            (3, 1): 3,  # dead    --B--> dead
        },
    }


def _build_bban_dfa() -> Dict:
    """
    Hand-constructed minimal DFA for bbaN: { B^n A^{2m} | n >= 1, m >= 0 }.

    Alphabet: A=0, B=1
    States:
      0 = q_start   (initial)
      1 = q_b       (in B-block, >= 1 B seen; ACCEPTING: n b's, 0 a's)
      2 = q_a_odd   (in A-block, odd # of A's)
      3 = q_a_even  (in A-block, even # of A's >= 2; ACCEPTING)
      4 = q_dead    (sink)

    Transitions:
      q_start  --B--> q_b         (first B)
      q_start  --A--> q_dead
      q_b      --B--> q_b         (more B's)
      q_b      --A--> q_a_odd     (first A)
      q_a_odd  --A--> q_a_even
      q_a_odd  --B--> q_dead
      q_a_even --A--> q_a_odd
      q_a_even --B--> q_dead
      q_dead   --A--> q_dead
      q_dead   --B--> q_dead
    """
    return {
        'states': [0, 1, 2, 3, 4],
        'alphabet': [0, 1],
        'initial': 0,
        'accepting': frozenset([1, 3]),
        'transitions': {
            (0, 0): 4,  # q_start  --A--> dead
            (0, 1): 1,  # q_start  --B--> q_b
            (1, 0): 2,  # q_b      --A--> q_a_odd
            (1, 1): 1,  # q_b      --B--> q_b
            (2, 0): 3,  # q_a_odd  --A--> q_a_even
            (2, 1): 4,  # q_a_odd  --B--> dead
            (3, 0): 2,  # q_a_even --A--> q_a_odd
            (3, 1): 4,  # q_a_even --B--> dead
            (4, 0): 4,  # dead     --A--> dead
            (4, 1): 4,  # dead     --B--> dead
        },
    }


def _get_dfa(grammar_name: str) -> Dict:
    """Return (or build and cache) the DFA for a grammar."""
    if grammar_name not in _DFA_CACHE:
        if grammar_name == 'baN':
            _DFA_CACHE[grammar_name] = _build_ban_dfa()
        elif grammar_name == 'bbaN':
            _DFA_CACHE[grammar_name] = _build_bban_dfa()
        else:
            raise ValueError(f"No DFA defined for grammar {grammar_name!r}")
    return _DFA_CACHE[grammar_name]


def _run_dfa(
    dfa: Dict, seq: Tuple[int, ...]
) -> Tuple[Set[int], Set[Tuple[int, int]]]:
    """
    Run a stripped sequence through the DFA, recording visited states and
    taken transitions.

    Args:
        dfa: DFA definition dict.
        seq: Stripped sequence as a tuple of token IDs.

    Returns:
        (visited_states, taken_transitions) where visited_states is a set of
        state IDs visited (including the initial) and taken_transitions is a
        set of (state, token) pairs that were traversed.
    """
    state: int = dfa['initial']
    visited: Set[int] = {state}
    taken: Set[Tuple[int, int]] = set()

    for token in seq:
        key = (state, int(token))
        next_state = dfa['transitions'].get(key)
        if next_state is None:
            break
        taken.add(key)
        state = next_state
        visited.add(state)

    return visited, taken


def compute_dfa_coverage(
    grammar_name: str, sequences: List[Tuple[int, ...]]
) -> Dict[str, float]:
    """
    Compute DFA state and transition coverage for baN or bbaN.

    Formula:
        dfa_state_coverage      = |states visited by any seq| / |total states|
        dfa_transition_coverage = |transitions taken by any seq| / |total transitions|

    Both metrics are bounded [0, 1]. A state is "visited" if any sequence in
    the batch passes through it. A transition is "taken" if any sequence
    triggers it.

    Args:
        grammar_name: 'baN' or 'bbaN'.
        sequences: List of stripped sequences.

    Returns:
        Dict with 'dfa_state_coverage' and 'dfa_transition_coverage'.
    """
    dfa = _get_dfa(grammar_name)
    all_visited: Set[int] = set()
    all_taken: Set[Tuple[int, int]] = set()

    for seq in sequences:
        vs, tt = _run_dfa(dfa, seq)
        all_visited.update(vs)
        all_taken.update(tt)

    n_states = len(dfa['states'])
    n_transitions = len(dfa['transitions'])

    return {
        'dfa_state_coverage': len(all_visited) / n_states,
        'dfa_transition_coverage': len(all_taken) / n_transitions,
    }


# ---------------------------------------------------------------------------
# n-distribution metrics (L3=aNbN, L5=aNbNcN)
# ---------------------------------------------------------------------------

def _entropy(values: List[int]) -> float:
    """
    Compute Shannon entropy H = -sum_k p_k log(p_k) in nats.

    A single-value distribution gives entropy 0.

    Args:
        values: List of integer observations.

    Returns:
        Float >= 0, or NaN if values is empty.
    """
    if not values:
        return float('nan')
    from collections import Counter
    counts = Counter(values)
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def compute_n_distribution(
    grammar_name: str,
    sequences: List[Tuple[int, ...]],
    vocab: Dict,
) -> Dict[str, float]:
    """
    Compute n-distribution metrics for aNbN (L3) and aNbNcN (L5).

    For each stripped sequence, n = count of A tokens (vocab['a']).

    Metrics:
      n_entropy: H(n) in nats. Single-value distribution → 0.
      n_coverage: |observed n values| / |valid n values|.

    The valid n range is read from vocab['valid_n_range'] (a range object),
    which must be set by the grammar's vocab_info() method.

    Args:
        grammar_name: 'aNbN' or 'aNbNcN'.
        sequences: List of stripped sequences.
        vocab: Dict from grammar.vocab_info(), must contain 'a' and 'valid_n_range'.

    Returns:
        Dict with 'n_entropy' and 'n_coverage'.
    """
    a_tok: int = vocab.get('a', 0)
    valid_range: range = vocab.get('valid_n_range', range(1, 2))

    n_values = [seq.count(a_tok) for seq in sequences]

    if not n_values:
        return {'n_entropy': float('nan'), 'n_coverage': float('nan')}

    h = _entropy(n_values)
    observed = set(n_values)
    valid_set = set(valid_range)
    coverage = len(observed) / len(valid_set) if valid_set else float('nan')

    return {'n_entropy': h, 'n_coverage': coverage}


# ---------------------------------------------------------------------------
# n,m-distribution metrics (L2=bbaN)
# ---------------------------------------------------------------------------

def compute_nm_distribution(
    sequences: List[Tuple[int, ...]], vocab: Dict
) -> Dict[str, float]:
    """
    Compute n,m-distribution metrics for bbaN (L2).

    For each stripped sequence:
      n = count of B tokens (vocab['b'])
      m = count of A tokens (vocab['a']) // 2

    Sequences with an odd A-count are skipped with a RuntimeWarning (they
    should not have been classified as correct).

    Metrics:
      n_entropy: H(n) in nats.
      m_entropy: H(m) in nats.
      nm_joint_coverage: |observed (n,m) pairs| / |valid (n,m) pairs|.

    The valid (n,m) pairs are read from vocab['valid_nm_pairs'] (a frozenset),
    which must be set by REGrammar.vocab_info() for bbaN.

    Args:
        sequences: List of stripped sequences.
        vocab: Dict with 'a', 'b', and 'valid_nm_pairs'.

    Returns:
        Dict with 'n_entropy', 'm_entropy', 'nm_joint_coverage'.
    """
    a_tok: int = vocab.get('a', 0)
    b_tok: int = vocab.get('b', 1)
    valid_nm: FrozenSet[Tuple[int, int]] = vocab.get('valid_nm_pairs', frozenset())

    n_values: List[int] = []
    m_values: List[int] = []
    nm_pairs: List[Tuple[int, int]] = []

    for seq in sequences:
        n_val = seq.count(b_tok)
        a_count = seq.count(a_tok)
        if a_count % 2 != 0:
            warnings.warn(
                f"bbaN sequence has odd A-count ({a_count}); "
                "it should not have been classified as correct. Skipping.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        m_val = a_count // 2
        n_values.append(n_val)
        m_values.append(m_val)
        nm_pairs.append((n_val, m_val))

    if not n_values:
        return {
            'n_entropy': float('nan'),
            'm_entropy': float('nan'),
            'nm_joint_coverage': float('nan'),
        }

    n_ent = _entropy(n_values)
    m_ent = _entropy(m_values)

    observed_nm = set(nm_pairs)
    joint_cov = (
        len(observed_nm) / len(valid_nm) if valid_nm else float('nan')
    )

    return {
        'n_entropy': n_ent,
        'm_entropy': m_ent,
        'nm_joint_coverage': joint_cov,
    }


# ---------------------------------------------------------------------------
# Dyck structure metrics (L4=parentheses_and_brackets, L6=not_nested_…)
# ---------------------------------------------------------------------------

def compute_dyck_structure(
    sequences: List[Tuple[int, ...]], vocab: Dict
) -> Dict[str, Any]:
    """
    Compute Dyck structure metrics for parentheses_and_brackets (L4) and
    not_nested_parentheses_and_brackets (L6).

    Per-sequence:

    max_depth_ratio:
        Treat every open token (+1) and every close token (-1) and compute the
        running prefix sum. max_depth_ratio = max(prefix_sums) / (L_content/2).
        Bound [0, 1] for valid Dyck-like words.

        Note: For L6 (CS Dyck), this metric conflates paren and bracket depths.
        This is intentional for cross-grammar comparability. If per-bracket-type
        depth is needed later, add separate metrics; for now, the combined version
        is the spec.

    brackets_parens_ratio:
        count(bracket_open) / count(paren_open) per sequence. Sequences with
        paren_open == 0 are excluded from the mean and counted separately in
        n_zero_paren_sequences.

    Scalar summaries:
      max_depth_ratio_mean, max_depth_ratio_std
      brackets_parens_ratio_mean, brackets_parens_ratio_std
      n_zero_paren_sequences

    Args:
        sequences: List of stripped sequences.
        vocab: Dict with 'paren_open', 'paren_close', 'bracket_open',
               'bracket_close'.

    Returns:
        Dict with the five scalar summaries above.
    """
    paren_open: int = vocab.get('paren_open', 0)
    paren_close: int = vocab.get('paren_close', 1)
    bracket_open: int = vocab.get('bracket_open', 6)
    bracket_close: int = vocab.get('bracket_close', 7)

    max_depth_ratios: List[float] = []
    bp_ratios: List[float] = []
    n_zero_paren = 0

    for seq in sequences:
        L_content = len(seq)
        if L_content == 0:
            continue

        # Max depth ratio: combined open/close depth
        depth = 0
        max_depth = 0
        for t in seq:
            if t == paren_open or t == bracket_open:
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif t == paren_close or t == bracket_close:
                depth -= 1
        ratio = max_depth / (L_content / 2)
        max_depth_ratios.append(ratio)

        # Brackets/parens ratio
        n_paren_open = seq.count(paren_open)
        n_bracket_open = seq.count(bracket_open)
        if n_paren_open == 0:
            n_zero_paren += 1
        else:
            bp_ratios.append(n_bracket_open / n_paren_open)

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float('nan')

    def _std(vals: List[float]) -> float:
        if len(vals) < 2:
            return float('nan')
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    return {
        'max_depth_ratio_mean': _mean(max_depth_ratios),
        'max_depth_ratio_std': _std(max_depth_ratios),
        'brackets_parens_ratio_mean': _mean(bp_ratios),
        'brackets_parens_ratio_std': _std(bp_ratios),
        'n_zero_paren_sequences': n_zero_paren,
    }


# ---------------------------------------------------------------------------
# Raw distributions for JSON sidefiles
# ---------------------------------------------------------------------------

def compute_diversity_distributions(
    grammar_name: str,
    correct_sequences: List[Any],
    vocab: Dict,
) -> Dict[str, Any]:
    """
    Compute raw distributions for JSON sidefiles.

    Sequences are stripped once before any computation.

    Args:
        grammar_name: Must be in GRAMMAR_METRIC_SETS; raises ValueError otherwise.
        correct_sequences: List of 1-D integer tensors or token-ID sequences
            satisfying rule1, rule2, and format.
        vocab: Dict from grammar.vocab_info().

    Returns:
        Dict with distribution data; schema depends on grammar:

        bbaN  → {"n_values": [...], "m_values": [...]}
        aNbN,
        aNbNcN → {"n_values": [...]}
        parentheses_and_brackets,
        not_nested_parentheses_and_brackets
              → {"max_depth_ratios": [...], "brackets_parens_ratios": [...],
                 "n_correct": N, "n_with_parens": M}
        baN   → {} (no distribution sidefile)
    """
    if grammar_name not in GRAMMAR_METRIC_SETS:
        raise ValueError(
            f"Unknown grammar {grammar_name!r}. "
            f"Known grammars: {sorted(GRAMMAR_METRIC_SETS.keys())}"
        )

    stripped = [_strip(seq, vocab) for seq in correct_sequences]
    metric_sets = GRAMMAR_METRIC_SETS[grammar_name]
    result: Dict[str, Any] = {}

    if 'nm_distribution' in metric_sets:
        # bbaN
        a_tok: int = vocab.get('a', 0)
        b_tok: int = vocab.get('b', 1)
        n_values: List[int] = []
        m_values: List[int] = []
        for seq in stripped:
            n_val = seq.count(b_tok)
            a_count = seq.count(a_tok)
            if a_count % 2 != 0:
                continue
            n_values.append(n_val)
            m_values.append(a_count // 2)
        result['n_values'] = n_values
        result['m_values'] = m_values

    elif 'n_distribution' in metric_sets:
        # aNbN, aNbNcN
        a_tok = vocab.get('a', 0)
        result['n_values'] = [seq.count(a_tok) for seq in stripped]

    if 'dyck_structure' in metric_sets:
        paren_open: int = vocab.get('paren_open', 0)
        paren_close: int = vocab.get('paren_close', 1)
        bracket_open: int = vocab.get('bracket_open', 6)
        bracket_close: int = vocab.get('bracket_close', 7)

        max_depth_ratios: List[float] = []
        bp_ratios: List[float] = []

        for seq in stripped:
            if not seq:
                continue
            L_content = len(seq)
            depth = 0
            max_depth = 0
            for t in seq:
                if t == paren_open or t == bracket_open:
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
                elif t == paren_close or t == bracket_close:
                    depth -= 1
            max_depth_ratios.append(max_depth / (L_content / 2))

            n_paren = seq.count(paren_open)
            n_bracket = seq.count(bracket_open)
            if n_paren > 0:
                bp_ratios.append(n_bracket / n_paren)

        result['max_depth_ratios'] = max_depth_ratios
        result['brackets_parens_ratios'] = bp_ratios
        result['n_correct'] = len(stripped)
        result['n_with_parens'] = len(bp_ratios)

    return result


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def compute_diversity_metrics(
    grammar_name: str,
    correct_sequences: List[Any],
    vocab: Dict,
    L: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute all applicable diversity metrics for a grammar.

    Sequences are stripped once at the top. Grammar-specific metrics are
    attempted for n_correct >= 1 (they may still return NaN if unsatisfiable).
    When n_correct == 0, grammar-specific metrics are all NaN.

    Args:
        grammar_name: Must be in GRAMMAR_METRIC_SETS; raises ValueError otherwise.
        correct_sequences: List of 1-D integer tensors or token-ID sequences
            satisfying rule1, rule2, and format.
        vocab: Dict from grammar.vocab_info(). Must include token IDs and any
            grammar-specific ranges (valid_n_range, valid_nm_pairs).
        L: Grammar content length (grammar.l). Unused directly; ranges are
           taken from vocab.

    Returns:
        Dict[str, Any] with all applicable metrics. Inapplicable metrics are
        NaN. Keys: n_correct, n_correct_too_low, uniqueness, duplication_rate,
        mean_lev_dist_normalized, lev_n_used, bigram_diversity, trigram_diversity,
        dfa_state_coverage, dfa_transition_coverage, n_entropy, n_coverage,
        m_entropy, nm_joint_coverage, max_depth_ratio_mean, max_depth_ratio_std,
        brackets_parens_ratio_mean, brackets_parens_ratio_std, n_zero_paren_sequences.

    Raises:
        ValueError: If grammar_name is not in GRAMMAR_METRIC_SETS.
    """
    if grammar_name not in GRAMMAR_METRIC_SETS:
        raise ValueError(
            f"Unknown grammar {grammar_name!r}. "
            f"Known grammars: {sorted(GRAMMAR_METRIC_SETS.keys())}"
        )

    metric_sets = GRAMMAR_METRIC_SETS[grammar_name]

    # Strip all sequences once at the top
    stripped = [_strip(seq, vocab) for seq in correct_sequences]
    n_correct = len(stripped)
    nan = float('nan')

    # --- Universal metrics ---
    result: Dict[str, Any] = _compute_universal(stripped)
    n_correct_too_low: bool = result['n_correct_too_low']

    # --- DFA coverage (baN, bbaN) ---
    if 'dfa_coverage' in metric_sets:
        if n_correct == 0:
            result['dfa_state_coverage'] = nan
            result['dfa_transition_coverage'] = nan
        else:
            result.update(compute_dfa_coverage(grammar_name, stripped))
    else:
        result['dfa_state_coverage'] = nan
        result['dfa_transition_coverage'] = nan

    # --- n-distribution (aNbN, aNbNcN) ---
    if 'n_distribution' in metric_sets:
        if n_correct == 0:
            result['n_entropy'] = nan
            result['n_coverage'] = nan
        else:
            result.update(compute_n_distribution(grammar_name, stripped, vocab))
        result['m_entropy'] = nan
        result['nm_joint_coverage'] = nan

    # --- nm-distribution (bbaN) ---
    elif 'nm_distribution' in metric_sets:
        if n_correct == 0:
            result['n_entropy'] = nan
            result['m_entropy'] = nan
            result['nm_joint_coverage'] = nan
        else:
            result.update(compute_nm_distribution(stripped, vocab))
        result['n_coverage'] = nan  # n_coverage only for L3/L5

    # --- neither n nor nm distribution (baN, Dyck grammars) ---
    else:
        result['n_entropy'] = nan
        result['n_coverage'] = nan
        result['m_entropy'] = nan
        result['nm_joint_coverage'] = nan

    # --- Dyck structure (parentheses_and_brackets, not_nested_…) ---
    if 'dyck_structure' in metric_sets:
        if n_correct == 0:
            result['max_depth_ratio_mean'] = nan
            result['max_depth_ratio_std'] = nan
            result['brackets_parens_ratio_mean'] = nan
            result['brackets_parens_ratio_std'] = nan
            result['n_zero_paren_sequences'] = 0
        else:
            result.update(compute_dyck_structure(stripped, vocab))
    else:
        result['max_depth_ratio_mean'] = nan
        result['max_depth_ratio_std'] = nan
        result['brackets_parens_ratio_mean'] = nan
        result['brackets_parens_ratio_std'] = nan
        result['n_zero_paren_sequences'] = nan

    return result
