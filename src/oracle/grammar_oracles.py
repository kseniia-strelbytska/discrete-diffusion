"""
Oracle marginal distributions for grammar-constrained sequences.

Project token scheme (constants.py):
  A=0, B=1, EOS=2, SOS=3, PAD=4, MASK=5, C=6
  open_paren=0, close_paren=1, open_bracket=6, close_bracket=7

Interface: each oracle function has signature
  fn(seq: Tensor[L], vocab_size: int) -> (status, Tensor[L, vocab_size]) | (None, str)
where status == 'expected_prob' on success.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from datasets.constants import *

# ─── shared helpers ───────────────────────────────────────────────────────────

def _validate(seq):
    """
    Return (True, eos_pos_or_None) if structural checks pass, else (False, msg).
    Checks: exactly one SOS at pos 0, at most one EOS, only PAD/MASK after EOS.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    if (seq == SOS_token).sum().item() != 1 or seq[0].item() != SOS_token:
        return False, 'SOS error'
    if (seq == EOS_token).sum().item() > 1:
        return False, 'Multiple EOS'
    eos_idx = (seq == EOS_token).nonzero(as_tuple=True)[0]
    eos_pos = eos_idx[0].item() if len(eos_idx) > 0 else None
    if eos_pos is not None:
        after = seq[eos_pos + 1:]
        if ((after != PAD_token) & (after != MASK_token)).any():
            return False, 'Non-PAD/MASK after EOS'
    return True, eos_pos


def _finish(counts_py, total, device):
    """
    Divide Python-native counts by total and build a float tensor.

    counts_py is a plain Python list-of-lists (shape L × vocab_size) whose
    values are arbitrary-precision Python ints.  Division happens entirely in
    Python before any torch conversion, so there is no int→C-long overflow
    regardless of how large the raw counts grow (e.g. 2^k for baN).
    """
    if total == 0:
        return None, 'No valid completion'
    L, V = len(counts_py), len(counts_py[0])
    probs = [[counts_py[pos][tok] / total for tok in range(V)] for pos in range(L)]
    return 'expected_prob', torch.tensor(probs, dtype=torch.float32, device=device)


def _check_pos_range(seq, positions, allowed):
    """Return True iff every position in positions has a token in allowed."""
    return all(seq[p].item() in allowed for p in positions)


# ─── aNbN oracle (delegates to existing implementation) ───────────────────────

def aNbN_get_marginals(seq, vocab_size=6):
    from oracle.deterministic_token_distribution import determineTokenDistribution
    return determineTokenDistribution(seq, vocab_size=vocab_size, device=seq.device)


# ─── baN oracle (L1) ──────────────────────────────────────────────────────────

def baN_get_marginals(seq, vocab_size=6):
    """
    Grammar: SOS B {A,B}^{n-1} EOS PAD*
    Total #A in content positions 1..n must be even.
    Position 1 is always B (contributes 0 A's), so #A in positions 2..n must be even.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]
    device = seq.device

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos
    if L > 1 and seq[1].item() not in (B, MASK_token):
        return None, 'Position 1 must be B'

    counts = [[0] * vocab_size for _ in range(L)]   # Python ints — no overflow
    total  = 0

    for n in range(1, L):          # content length n >= 1
        ep = n + 1                 # EOS position
        if ep >= L:
            break
        if eos_pos is not None and eos_pos != ep:
            continue
        if seq[ep].item() not in (EOS_token, MASK_token):
            continue
        if not _check_pos_range(seq, range(ep + 1, L), (PAD_token, MASK_token)):
            continue
        if not _check_pos_range(seq, range(2, n + 1), (A, B, MASK_token)):
            continue

        rev_a = sum(1 for p in range(2, n + 1) if seq[p].item() == A)
        k     = sum(1 for p in range(2, n + 1) if seq[p].item() == MASK_token)
        # need (rev_a + mask_a) even → mask_a must have parity = rev_a % 2
        parity = rev_a % 2
        cnt = (1 if parity == 0 else 0) if k == 0 else 2 ** (k - 1)
        if cnt == 0:
            continue

        total                   += cnt
        counts[0] [SOS_token]   += cnt
        counts[1] [B]           += cnt
        counts[ep][EOS_token]   += cnt
        for p in range(ep + 1, L):
            counts[p][PAD_token] += cnt

        for p in range(2, n + 1):
            t = seq[p].item()
            if t == A:
                counts[p][A] += cnt
            elif t == B:
                counts[p][B] += cnt
            else:  # MASK
                if k == 1:
                    ca = 1 if (parity + 1) % 2 == 0 else 0   # i.e., 1 iff parity==1
                    cb = 1 if parity == 0 else 0
                else:
                    ca = cb = 2 ** (k - 2)
                counts[p][A] += ca
                counts[p][B] += cb

    return _finish(counts, total, device)


# ─── bbaN oracle (L2) ─────────────────────────────────────────────────────────

def bbaN_get_marginals(seq, vocab_size=6):
    """
    Optimized O(L^2) Grammar Oracle for: SOS B^n A^{2m} EOS PAD* (n >= 1, m >= 0)
    Uses precomputed validation prefixes and a difference array sweep to achieve maximum efficiency.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]
    device = seq.device

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    # 1. Cache lookups and precompute scalar checks to completely avoid .item() inside loops
    seq_items = seq.tolist()
    can_B = [t in (B, MASK_token) for t in seq_items]
    can_A = [t in (A, MASK_token) for t in seq_items]
    can_PAD = [t in (PAD_token, MASK_token) for t in seq_items]
    can_EOS = [t in (EOS_token, MASK_token) for t in seq_items]

    # Cumulative prefix validation for B: valid_B_prefix[n] is True iff 1..n can all be B
    valid_B_prefix = [True] * L
    for p in range(1, L):
        valid_B_prefix[p] = valid_B_prefix[p - 1] and can_B[p]

    # Cumulative suffix validation for PAD: valid_PAD_suffix[ep] is True iff ep+1..L-1 can all be PAD
    valid_PAD_suffix = [True] * (L + 1)
    for p in range(L - 2, -1, -1):
        valid_PAD_suffix[p] = valid_PAD_suffix[p + 1] and can_PAD[p + 1]

    # 2. Initialize a difference array for O(1) interval modifications
    diff = [[0] * vocab_size for _ in range(L + 1)]
    total = 0

    # Iterate over n (B-boundary) and ep (EOS position). 2m is implicitly defined as (ep - n - 1)
    for n in range(1, L - 1):
        if not valid_B_prefix[n]:
            break  # If prefix 1..n cannot be B, then 1..n+1 definitely cannot be B. Exit early!
        
        A_block_valid = True
        for ep in range(n + 1, L):
            if ep - 1 > n:
                A_block_valid = A_block_valid and can_A[ep - 1]
            
            if not A_block_valid:
                break  # If the consecutive A block breaks, extending ep further won't fix it. Exit early!
            
            # Structural constraint filtering
            if eos_pos is not None and eos_pos != ep:
                continue
            if not can_EOS[ep]:
                continue
            if not valid_PAD_suffix[ep]:
                continue
            if (ep - n - 1) % 2 != 0:  # Length of A block (2m) must be even
                continue

            # We found a perfectly valid world!
            total += 1
            
            # Perform O(1) interval updates on our difference array
            diff[0][SOS_token]   += 1
            diff[1][SOS_token]   -= 1
            
            diff[1][B]           += 1
            diff[n + 1][B]       -= 1
            
            if ep > n + 1:
                diff[n + 1][A]   += 1
                diff[ep][A]       -= 1
                
            diff[ep][EOS_token]  += 1
            diff[ep + 1][EOS_token] -= 1
            
            if ep + 1 < L:
                diff[ep + 1][PAD_token] += 1
                diff[L][PAD_token]      -= 1

    # 3. Reconstruct final token counts via a single prefix-sum sweep
    if total == 0:
        return None, 'No valid completion'

    counts = [[0] * vocab_size for _ in range(L)]
    current_counts = [0] * vocab_size
    for p in range(L):
        for t in range(vocab_size):
            current_counts[t] += diff[p][t]
            counts[p][t] = current_counts[t]

    return _finish(counts, total, device)


# ─── aNbNcN oracle (L5) ───────────────────────────────────────────────────────

def aNbNcN_get_marginals(seq, vocab_size=7):
    """
    Grammar: SOS A^n B^n C^n EOS PAD*   (n >= 1)
    Each n determines the sequence completely.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]
    device = seq.device

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    counts = [[0] * vocab_size for _ in range(L)]   # Python ints — no overflow
    total  = 0

    for n in range(1, L):
        ep = 3 * n + 1
        if ep >= L:
            break
        if eos_pos is not None and eos_pos != ep:
            continue
        if seq[ep].item() not in (EOS_token, MASK_token):
            continue
        if not _check_pos_range(seq, range(ep + 1, L), (PAD_token, MASK_token)):
            continue
        if not _check_pos_range(seq, range(1,         n + 1),     (A, MASK_token)):
            continue
        if not _check_pos_range(seq, range(n + 1,     2 * n + 1), (B, MASK_token)):
            continue
        if not _check_pos_range(seq, range(2 * n + 1, 3 * n + 1), (C, MASK_token)):
            continue

        total                 += 1
        counts[0] [SOS_token] += 1
        for p in range(1,         n + 1):      counts[p][A] += 1
        for p in range(n + 1,     2 * n + 1):  counts[p][B] += 1
        for p in range(2 * n + 1, 3 * n + 1):  counts[p][C] += 1
        counts[ep][EOS_token] += 1
        for p in range(ep + 1, L):
            counts[p][PAD_token] += 1

    return _finish(counts, total, device)


# ─── Dyck DP (shared forward–backward engine) ─────────────────────────────────

def _dyck_forward(seq, n, content_tokens, trans_fn):
    """
    Forward DP over content positions 1..n.
    fwd[pos] = {state: count}; fwd[0] = initial state with count 1.
    trans_fn(state, token) -> new_state or None if invalid.
    """
    fwd = [None] * (n + 1)
    fwd[0] = {_initial_state(trans_fn): 1}
    for pos in range(1, n + 1):
        t_seq = seq[pos].item()
        toks = content_tokens if t_seq == MASK_token else [t_seq]
        nf = {}
        for t in toks:
            for s, c in fwd[pos - 1].items():
                ns = trans_fn(s, t)
                if ns is not None:
                    nf[ns] = nf.get(ns, 0) + c
        fwd[pos] = nf
    return fwd


def _dyck_backward(seq, n, content_tokens, trans_fn, inv_trans_fn, final_state):
    """
    Backward DP: bwd[pos][s] = ways to complete positions pos+1..n from state s to final_state.
    """
    bwd = [None] * (n + 1)
    bwd[n] = {final_state: 1}
    for pos in range(n - 1, -1, -1):
        t_next = seq[pos + 1].item()
        toks = content_tokens if t_next == MASK_token else [t_next]
        nb = {}
        for sp, bc in bwd[pos + 1].items():
            for t in toks:
                for s in inv_trans_fn(sp, t):
                    nb[s] = nb.get(s, 0) + bc
        bwd[pos] = nb
    return bwd


def _initial_state(trans_fn):
    """Sentinel: derive initial state tag from trans_fn's closure."""
    # Caller sets initial state explicitly; this is a placeholder.
    return None  # overridden below


def _dyck_oracle(seq, vocab_size, content_tokens, trans_fn, inv_trans_fn, final_state):
    """
    Generic Dyck oracle.  Content length must be even and >= 2.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]
    device = seq.device

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    counts = [[0] * vocab_size for _ in range(L)]   # Python ints — no overflow
    total  = 0

    for n in range(2, L - 1, 2):   # even content length >= 2
        ep = n + 1
        if ep >= L:
            break
        if eos_pos is not None and eos_pos != ep:
            continue
        if seq[ep].item() not in (EOS_token, MASK_token):
            continue
        if not _check_pos_range(seq, range(ep + 1, L), (PAD_token, MASK_token)):
            continue
        if not _check_pos_range(seq, range(1, n + 1), content_tokens + [MASK_token]):
            continue

        fwd = _dyck_forward_impl(seq, n, content_tokens, trans_fn, final_state)
        cnt = fwd[n].get(final_state, 0)
        if cnt == 0:
            continue

        bwd = _dyck_backward_impl(seq, n, content_tokens, inv_trans_fn, final_state)

        total                 += cnt
        counts[0] [SOS_token] += cnt
        counts[ep][EOS_token] += cnt
        for p in range(ep + 1, L):
            counts[p][PAD_token] += cnt

        for pos in range(1, n + 1):
            t_seq = seq[pos].item()
            if t_seq != MASK_token:
                counts[pos][t_seq] += cnt
            else:
                for t in content_tokens:
                    tc = 0
                    for s, fc in fwd[pos - 1].items():
                        ns = trans_fn(s, t)
                        if ns is not None:
                            tc += fc * bwd[pos].get(ns, 0)
                    counts[pos][t] += tc

    return _finish(counts, total, device)


def _dyck_forward_impl(seq, n, content_tokens, trans_fn, initial_state):
    fwd = [None] * (n + 1)
    fwd[0] = {initial_state: 1}
    for pos in range(1, n + 1):
        t_seq = seq[pos].item()
        toks = content_tokens if t_seq == MASK_token else [t_seq]
        nf = {}
        for t in toks:
            for s, c in fwd[pos - 1].items():
                ns = trans_fn(s, t)
                if ns is not None:
                    nf[ns] = nf.get(ns, 0) + c
        fwd[pos] = nf
    return fwd


def _dyck_backward_impl(seq, n, content_tokens, inv_trans_fn, final_state):
    bwd = [None] * (n + 1)
    bwd[n] = {final_state: 1}
    for pos in range(n - 1, -1, -1):
        t_next = seq[pos + 1].item()
        toks = content_tokens if t_next == MASK_token else [t_next]
        nb = {}
        for sp, bc in bwd[pos + 1].items():
            for t in toks:
                for s in inv_trans_fn(sp, t):
                    nb[s] = nb.get(s, 0) + bc
        bwd[pos] = nb
    return bwd


# ─── not_nested_parentheses_and_brackets oracle (L6) ──────────────────────────

# State: (depth_paren, depth_bracket)
_IND_INITIAL = (0, 0)
_IND_FINAL   = (0, 0)
_IND_TOKENS  = [OPEN_P, CLOSE_P, OPEN_B, CLOSE_B]


def _ind_trans(state, t):
    da, db = state
    if t == OPEN_P:  return (da + 1, db)
    if t == CLOSE_P: return (da - 1, db) if da > 0 else None
    if t == OPEN_B:  return (da, db + 1)
    if t == CLOSE_B: return (da, db - 1) if db > 0 else None
    return None


def _ind_inv_trans(sp, t):
    """Yield all states s such that _ind_trans(s, t) == sp."""
    da_p, db_p = sp
    if t == OPEN_P:
        if da_p >= 1: yield (da_p - 1, db_p)
    elif t == CLOSE_P:
        yield (da_p + 1, db_p)
    elif t == OPEN_B:
        if db_p >= 1: yield (da_p, db_p - 1)
    elif t == CLOSE_B:
        yield (da_p, db_p + 1)


def not_nested_parentheses_and_brackets_get_marginals(seq, vocab_size=8):
    """
    Grammar: each of the paren-subsequence and bracket-subsequence is
    independently a valid Dyck-1 string (they can interleave freely).
    """
    return _dyck_oracle(
        seq, vocab_size,
        content_tokens=_IND_TOKENS,
        trans_fn=_ind_trans,
        inv_trans_fn=_ind_inv_trans,
        final_state=_IND_FINAL,
    )


# ─── parentheses_and_brackets oracle (L4) — interval DP ──────────────────────
#
# Previous implementation used a tuple-as-stack DP state, giving 2^(n/2)
# reachable states for a masked sequence of content length n — exponential.
#
# This implementation uses interval DP.
# dp[(i, j)] = number of valid Dyck-2 completions of seq[i..j].
# Recurrence: the bracket that opens at position i must close at some k in
#   {i+1, i+3, ...} (odd offset so the inner span has even length).
#   dp[(i,j)] = sum_{k, bracket_type} dp[(i+1, k-1)] * dp[(k+1, j)]
# Complexity: O(n^3) for the DP table, O(n^4) for all marginals via
#   pin-and-rerun, O(L^5) worst-case over all candidate content lengths.
# For L=20 this is ~3 * 10^6 operations — fully tractable.

_DYCK2_PAIRS   = ((OPEN_P, CLOSE_P), (OPEN_B, CLOSE_B))
_DYCK2_CONTENT = (OPEN_P, CLOSE_P, OPEN_B, CLOSE_B)


def _dyck2_dp(seq_list, start, end):
    """
    Count valid Dyck-2 completions for seq_list[start..end] (inclusive).
    Returns 1 for empty interval (start > end), 0 for odd-length interval.
    """
    if start > end:
        return 1
    if (end - start + 1) % 2 == 1:
        return 0
    dp = {}
    for length in range(0, end - start + 2, 2):   # 0, 2, 4, … (even only)
        for i in range(start, end - length + 2):
            j = i + length - 1                     # j < i when length == 0
            if length == 0:
                dp[(i, j)] = 1                      # empty span → 1 completion
                continue
            total = 0
            for k in range(i + 1, j + 1, 2):       # k - i is odd
                si, sk = seq_list[i], seq_list[k]
                for open_tok, close_tok in _DYCK2_PAIRS:
                    if si not in (open_tok, MASK_token):
                        continue
                    if sk not in (close_tok, MASK_token):
                        continue
                    total += dp[(i + 1, k - 1)] * dp[(k + 1, j)]
            dp[(i, j)] = total
    return dp[(start, end)]


def _dyck2_dp_pinned(seq_list, start, end, pin_pos, pin_val):
    """Run _dyck2_dp with seq_list[pin_pos] temporarily set to pin_val."""
    orig = seq_list[pin_pos]
    seq_list[pin_pos] = pin_val
    result = _dyck2_dp(seq_list, start, end)
    seq_list[pin_pos] = orig
    return result


def parentheses_and_brackets_get_marginals(seq, vocab_size=8):
    """
    Grammar: SOS [Dyck-2 word of even length n] EOS PAD*

    Uses interval DP: O(n^4) per candidate content length n.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]
    device = seq.device

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    seq_list = seq.tolist()
    counts    = [[0] * vocab_size for _ in range(L)]
    grand_total = 0

    for n in range(2, L - 1, 2):      # even content length ≥ 2 (at least one pair)
        ep = n + 1                      # EOS position
        if ep >= L:
            break
        if eos_pos is not None and eos_pos != ep:
            continue
        if seq_list[ep] not in (EOS_token, MASK_token):
            continue
        if not _check_pos_range(seq, range(ep + 1, L), (PAD_token, MASK_token)):
            continue
        if n > 0 and not _check_pos_range(
            seq, range(1, n + 1), (OPEN_P, CLOSE_P, OPEN_B, CLOSE_B, MASK_token)
        ):
            continue

        total_n = _dyck2_dp(seq_list, 1, n)
        if total_n == 0:
            continue

        grand_total           += total_n
        counts[0][SOS_token]  += total_n
        counts[ep][EOS_token] += total_n
        for p in range(ep + 1, L):
            counts[p][PAD_token] += total_n

        for pos in range(1, n + 1):
            t = seq_list[pos]
            if t != MASK_token:
                counts[pos][t] += total_n          # unmasked: deterministic
            else:
                for v in _DYCK2_CONTENT:           # masked: pin-and-rerun
                    counts[pos][v] += _dyck2_dp_pinned(seq_list, 1, n, pos, v)

    return _finish(counts, grand_total, device)


# ─── unified dispatch ─────────────────────────────────────────────────────────

_ORACLE_MAP = {
    'aNbN':                              aNbN_get_marginals,
    'baN':                               baN_get_marginals,
    'bbaN':                              bbaN_get_marginals,
    'aNbNcN':                            aNbNcN_get_marginals,
    'not_nested_parentheses_and_brackets': not_nested_parentheses_and_brackets_get_marginals,
    'parentheses_and_brackets':          parentheses_and_brackets_get_marginals,
}

_VOCAB_SIZE_MAP = {
    'aNbN': 6, 'baN': 6, 'bbaN': 6,
    'aNbNcN': 7,
    'not_nested_parentheses_and_brackets': 8,
    'parentheses_and_brackets': 8,
}


def get_marginals(grammar_name, seq, vocab_size=None):
    """
    Unified entry point.

    Args:
        grammar_name: string key from _ORACLE_MAP (or 'anbn' alias for 'aNbN')
        seq: 1-D LongTensor of length L
        vocab_size: if None, inferred from _VOCAB_SIZE_MAP; always floored to the
                    grammar's minimum required size so token indices never go OOB.

    Returns:
        ('expected_prob', Tensor[L, vocab_size])  or  (None, error_string)
    """
    resolved = 'aNbN' if grammar_name == 'anbn' else grammar_name
    if resolved not in _ORACLE_MAP:
        return None, f'No oracle implemented for grammar {grammar_name!r}'
    min_vocab = _VOCAB_SIZE_MAP.get(resolved, 0)
    if vocab_size is None:
        vocab_size = min_vocab
    else:
        vocab_size = max(vocab_size, min_vocab)
    return _ORACLE_MAP[resolved](seq, vocab_size)


# ─── nn.Module wrapper ────────────────────────────────────────────────────────

class oracleModel(nn.Module):
    """
    Drop-in replacement for the aNbN-only oracle in deterministic_token_distribution.py.
    Routes each forward() call to the correct grammar's oracle via get_marginals().

    Args:
        grammar_name: grammar key (e.g. 'anbn', 'baN', 'aNbNcN', …)
        vocab_size:   embedding-table size for the current grammar
        device:       torch device
    """

    def __init__(self, grammar_name: str, vocab_size: int, device):
        super().__init__()
        self.grammar_name = grammar_name
        resolved = 'aNbN' if grammar_name == 'anbn' else grammar_name
        self.vocab_size = max(vocab_size, _VOCAB_SIZE_MAP.get(resolved, 0))
        self.device = device
        self.architecture = 'diffusion'
        self.oracle = True

    def forward(self, X, timestep=None):
        """
        Args:
            X: (L,) or (1, L) LongTensor — partially masked sequence
        Returns:
            (L, vocab_size) or (1, L, vocab_size) FloatTensor of marginals
        """
        X = X.to(self.device)
        squeeze = X.ndim == 1
        if not squeeze:
            X = X.view(-1)

        status, result = get_marginals(self.grammar_name, X.cpu(), self.vocab_size)
        if status is None:
            raise ValueError(result)

        result = result.to(self.device)
        return result if squeeze else result.unsqueeze(0)
