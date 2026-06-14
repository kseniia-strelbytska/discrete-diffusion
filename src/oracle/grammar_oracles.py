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
from datasets.constants import SOS_token, EOS_token, PAD_token, MASK_token

# Content token IDs
A = 0; B = 1; C = 6
OPEN_P = 0; CLOSE_P = 1   # parentheses
OPEN_B = 6; CLOSE_B = 7   # brackets


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


def _finish(counts, total):
    """Normalise counts or return None if no completions found."""
    if total == 0:
        return None, 'No valid completion'
    return 'expected_prob', (counts / total).float()


def _check_pos_range(seq, positions, allowed):
    """Return True iff every position in positions has a token in allowed."""
    return all(seq[p].item() in allowed for p in positions)


# ─── aNbN oracle (delegates to existing implementation) ───────────────────────

def aNbN_get_marginals(seq, vocab_size=6):
    from oracle.deterministic_token_distribution import determineTokenDistribution
    return determineTokenDistribution(seq, vocab_size=vocab_size, device='cpu')


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

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos
    if L > 1 and seq[1].item() not in (B, MASK_token):
        return None, 'Position 1 must be B'

    counts = torch.zeros(L, vocab_size, dtype=torch.float64)
    total = 0.0

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

        total += cnt
        counts[0,  SOS_token] += cnt
        counts[1,  B]         += cnt
        counts[ep, EOS_token] += cnt
        for p in range(ep + 1, L):
            counts[p, PAD_token] += cnt

        for p in range(2, n + 1):
            t = seq[p].item()
            if t == A:
                counts[p, A] += cnt
            elif t == B:
                counts[p, B] += cnt
            else:  # MASK
                if k == 1:
                    # fixing to A: remaining 0 masked pos need parity (parity+1)%2 met by 0 items
                    ca = 1 if (parity + 1) % 2 == 0 else 0   # i.e., 1 iff parity==1
                    cb = 1 if parity == 0 else 0
                else:
                    ca = cb = 2 ** (k - 2)
                counts[p, A] += ca
                counts[p, B] += cb

    return _finish(counts, total)


# ─── bbaN oracle (L2) ─────────────────────────────────────────────────────────

def bbaN_get_marginals(seq, vocab_size=6):
    """
    Grammar: SOS B^n A^{2m} EOS PAD*   (n >= 1, m >= 0)
    Each (n, m) pair determines the sequence completely.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    counts = torch.zeros(L, vocab_size, dtype=torch.float64)
    total = 0.0

    for n in range(1, L):             # B count >= 1
        for m in range(0, L // 2 + 1):
            content_len = n + 2 * m
            ep = content_len + 1
            if ep >= L:
                break
            if eos_pos is not None and eos_pos != ep:
                continue
            if seq[ep].item() not in (EOS_token, MASK_token):
                continue
            if not _check_pos_range(seq, range(ep + 1, L), (PAD_token, MASK_token)):
                continue
            if not _check_pos_range(seq, range(1, n + 1), (B, MASK_token)):
                continue
            if not _check_pos_range(seq, range(n + 1, n + 2 * m + 1), (A, MASK_token)):
                continue

            # exactly 1 completion for this (n, m)
            total += 1
            counts[0, SOS_token] += 1
            for p in range(1, n + 1):
                counts[p, B] += 1
            for p in range(n + 1, n + 2 * m + 1):
                counts[p, A] += 1
            counts[ep, EOS_token] += 1
            for p in range(ep + 1, L):
                counts[p, PAD_token] += 1

    return _finish(counts, total)


# ─── aNbNcN oracle (L5) ───────────────────────────────────────────────────────

def aNbNcN_get_marginals(seq, vocab_size=7):
    """
    Grammar: SOS A^n B^n C^n EOS PAD*   (n >= 1)
    Each n determines the sequence completely.
    """
    if seq.ndim != 1:
        seq = seq.view(-1)
    L = seq.shape[0]

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    counts = torch.zeros(L, vocab_size, dtype=torch.float64)
    total = 0.0

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

        total += 1
        counts[0, SOS_token] += 1
        for p in range(1,         n + 1):     counts[p, A] += 1
        for p in range(n + 1,     2 * n + 1): counts[p, B] += 1
        for p in range(2 * n + 1, 3 * n + 1): counts[p, C] += 1
        counts[ep, EOS_token] += 1
        for p in range(ep + 1, L):
            counts[p, PAD_token] += 1

    return _finish(counts, total)


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

    ok, eos_pos = _validate(seq)
    if not ok:
        return None, eos_pos

    counts = torch.zeros(L, vocab_size, dtype=torch.float64)
    total = 0.0

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

        total += cnt
        counts[0, SOS_token] += cnt
        counts[ep, EOS_token] += cnt
        for p in range(ep + 1, L):
            counts[p, PAD_token] += cnt

        for pos in range(1, n + 1):
            t_seq = seq[pos].item()
            if t_seq != MASK_token:
                counts[pos, t_seq] += cnt
            else:
                for t in content_tokens:
                    tc = 0
                    for s, fc in fwd[pos - 1].items():
                        ns = trans_fn(s, t)
                        if ns is not None:
                            tc += fc * bwd[pos].get(ns, 0)
                    counts[pos, t] += tc

    return _finish(counts, total)


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


# ─── parentheses_and_brackets oracle (L4) ─────────────────────────────────────

# State: stack as a tuple of open bracket tokens (nested Dyck)
_NEST_INITIAL = ()
_NEST_FINAL   = ()
_NEST_TOKENS  = [OPEN_P, CLOSE_P, OPEN_B, CLOSE_B]
_OPEN_TO_CLOSE = {OPEN_P: CLOSE_P, OPEN_B: CLOSE_B}
_CLOSE_TO_OPEN = {CLOSE_P: OPEN_P, CLOSE_B: OPEN_B}


def _nest_trans(state, t):
    if t in _OPEN_TO_CLOSE:
        return state + (t,)
    if t in _CLOSE_TO_OPEN:
        exp = _CLOSE_TO_OPEN[t]
        if not state or state[-1] != exp:
            return None
        return state[:-1]
    return None


def _nest_inv_trans(sp, t):
    """Yield all states s such that _nest_trans(s, t) == sp."""
    if t in _OPEN_TO_CLOSE:
        # sp = s + (t,)  →  s = sp[:-1] if sp[-1] == t
        if sp and sp[-1] == t:
            yield sp[:-1]
    elif t in _CLOSE_TO_OPEN:
        exp = _CLOSE_TO_OPEN[t]
        # sp = s[:-1] and s[-1] == exp  →  s = sp + (exp,)
        yield sp + (exp,)


def parentheses_and_brackets_get_marginals(seq, vocab_size=8):
    """
    Grammar: properly nested () and [] (standard bracket matching).
    """
    return _dyck_oracle(
        seq, vocab_size,
        content_tokens=_NEST_TOKENS,
        trans_fn=_nest_trans,
        inv_trans_fn=_nest_inv_trans,
        final_state=_NEST_FINAL,
    )


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
        squeeze = X.ndim == 1
        if not squeeze:
            X = X.view(-1)

        status, result = get_marginals(self.grammar_name, X.cpu(), self.vocab_size)
        if status is None:
            raise ValueError(result)

        result = result.to(self.device)
        return result if squeeze else result.unsqueeze(0)
