import torch 
import torch.nn as nn
import random
from tqdm import tqdm
from itertools import product
import numpy as np
from constants import SOS_token, EOS_token, PAD_token, MASK_token
from noise_schedule_unmask import ScheduledUnmasker

class oracleModel(nn.Module):
    def __init__(self, vocab_size, device):
        super(oracleModel, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.architecture = 'diffusion'
        self.oracle = True
    
    def forward(self, X):
        # X shape: (L,)
        # output shape: (L, vocab_size)
        pred = determineTokenDistribution(X, vocab_size=self.vocab_size, device=self.device)
        return pred


def determineTokenDistribution(seq, vocab_size, device):
    '''
    seq: partially masked sequence
    
    What if there are multiple SOS?
    What to do if no valid completion exists?
    '''
    
    if seq.ndim != 1:
        seq = seq.view(max(seq.shape))
    
    if (seq == SOS_token).sum() != 1 or seq[0] != SOS_token:
        return (None, f'Error: Sequence contains {(seq == SOS_token).sum()} SOS tokens.')
    
    if (seq == EOS_token).sum() > 1:
        return (None, f'Error: Sequence contains {(seq == EOS_token).sum()} EOS tokens.')
    
    EOS_pos = (seq == EOS_token).nonzero(as_tuple=True)[0][0] if (seq == EOS_token).sum() > 0 else None # earliest possible EOS
    expected_prob = torch.zeros((seq.shape[0], vocab_size), device=device)
    
    pad_first = (seq == PAD_token).nonzero(as_tuple=True)[0][0] if (seq == PAD_token).sum() > 0 else seq.shape[0] # earliest present pad
    zero_last = (seq == 0).nonzero(as_tuple=True)[0][-1] if (seq == 0).sum() > 0 else None # latest present zero
    one_first = (seq == 1).nonzero(as_tuple=True)[0][0] if (seq == 1).sum() > 0 else None # earliest present one
    one_last = (seq == 1).nonzero(as_tuple=True)[0][-1] if (seq == 1).sum() > 0 else None # latest present one
        
    if zero_last is not None and one_first is not None and zero_last > one_first:
        return (None, f'Error: The sequence cannot be completed correctly. The sequence contains no EOS, and there is a zero at position {zero_last} that is after a one at position {one_first}')  
    
    if pad_first + 1 < seq.shape[0] and ((seq[pad_first + 1:] == 0) | (seq[pad_first + 1:] == 1)).sum() > 0:
        return (None, f'Error: The sequence cannot be completed correctly. The sequence contains no EOS, and there is a PAD token at position {pad_first} that is followed by non-PAD tokens.')
    
    if EOS_pos is not None:
        # EOS is present => one possible valid completion 
        if EOS_pos % 2 == 0:
            return (None, f'Error: The sequence cannot be completed correctly. The sequence contains EOS and there is an odd number of tokens between SOS and EOS')
        
        if EOS_pos + 1 < seq.shape[0] and ((seq[EOS_pos + 1:] == 0) | (seq[EOS_pos + 1:] == 1)).sum() > 0:
            return (None, f'Error: The sequence cannot be completed correctly. The sequence contains EOS, and there are non-PAD tokens after EOS.')
        
        l = EOS_pos // 2
        
        if l == 0:
            return (None, f'Error: The sequence cannot be completed correctly. The sequence contains EOS, but there are no tokens between SOS and EOS. This sample is not part of the training data, therefore is not considered a valid completion.')
        
        # 1...l: '0', l+1...2l: '1'
        if ((seq[1 : l+1]==1) | (seq[1 : l+1]==PAD_token)).sum() > 0 or \
           ((seq[l+1 : 2*l+1] == 0) | (seq[l+1 : 2*l+1] == PAD_token)).sum() > 0:
            return (None, f'Error: The sequence cannot be completed correctly. The sequence contains EOS, and there are {(seq[1 : l+1]==1).sum()} 1s in the left half and {(seq[l+1 : 2*l+1] == 1).sum()} 0s in the right half')
        
        # deterministic
        expected_prob[0, SOS_token] = 1.0
        expected_prob[1 : l+1, 0] = 1.0
        expected_prob[l + 1 : 2 * l + 1, 1] = 1.0
        expected_prob[EOS_pos, EOS_token] = 1.0
        
        if EOS_pos + 1 < seq.shape[0]:
            expected_prob[EOS_pos + 1:, PAD_token] = 1.0
        
        return ('expected_prob', expected_prob)
    
    # no EOS => multiple valid completions
    lower_bound, upper_bound = 1, None
    
    upper_bound = (pad_first - 2) // 2 # need to have space for EOS before PAD
    if one_first is not None:
        upper_bound = min(upper_bound, one_first - 1) # the last zero must be before the first one
    
    if zero_last is not None:
        lower_bound = max(lower_bound, zero_last) # need to cover the last zero
    if one_last is not None:
        lower_bound = max(lower_bound, (one_last + 1) // 2) # need to cover the last one
        
    if lower_bound > upper_bound:
        return (None, f'Error: The sequence cannot be completed correctly. The sequence contains no EOS, and the constraints on the possible completions are contradictory: \nThe lower bound on the number of zeros is {lower_bound} and the upper bound is {upper_bound}.')
    
    total = upper_bound - lower_bound + 1
    if total == 1:
        expected_prob[0, SOS_token] = 1.0
        expected_prob[1 : lower_bound + 1, 0] = 1.0
        expected_prob[lower_bound + 1 : 2 * lower_bound + 1, 1] = 1.0
        expected_prob[2 * lower_bound + 1, EOS_token] = 1.0
        
        if 2 * lower_bound + 2 < seq.shape[0]:
            expected_prob[2 * lower_bound + 2:, PAD_token] = 1.0
        return ('expected_prob', expected_prob)
    
    expected_prob[0, SOS_token] = total
    for i in range(lower_bound, upper_bound + 1):
        expected_prob[1:i+1, 0] += 1
        expected_prob[i + 1 : 2 * i + 1, 1] += 1
        expected_prob[2 * i + 1, EOS_token] += 1
        
        if 2 * i + 2 < seq.shape[0]:
            expected_prob[2 * i + 2:, PAD_token] += 1
    
    if expected_prob.sum(-1).min() != total or expected_prob.sum(-1).max() != total:
        print('Error in implementation! This should not happen. The expected probabilities should sum to the total number of valid completions for each token position.')
        exit(1)
    
    expected_prob /= total
    return ('expected_prob', expected_prob)
    
    '''
    Efficient version:
    
    Example:
    seq.shape = (10,)
    idxs: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    lower_bound = 2, upper_bound = 4 => total = 3
    
    The number of valid completions with zero at token i:
    0, 3, 3, 2, 1, 0, 0, 0, 0, 0
    The number of valid completions with one at token i:
    0, 0, 0, 1, 2, 2, 2, 1, 1, 0
    
    zero_decr: tensor([2, 1]) 
    
    '''
    # zero_decr = torch.flip(torch.arange(1, total - 1), dims=[0])
    # expected_prob[0][SOS_token] = total
    # expected_prob[1 : lower_bound + 1][0] = total
    # expected_prob[lower_bound+1:upper_bound+1][0] = zero_decr

def check_completion(seq, completion):
    for i in range(seq.shape[0]):
        if seq[i] != completion[i]:
            if seq[i] != MASK_token:
                return False 
            
    return True

def correct_determineTokenDistribution(seq, vocab_size, device):
    if seq.ndim != 1:
        seq = seq.view(max(seq.shape))
    
    expected_prob = torch.zeros((seq.shape[0], vocab_size))
    total = 0

    for l in range(1, seq.shape[0] // 2):
        completion = torch.tensor([SOS_token] + l * [0] + l * [1] + [EOS_token] + max(0, int(seq.shape[0]) - 2 * l - 2) * [PAD_token], dtype=torch.long)
        
        if completion.shape[0] == seq.shape[0] and check_completion(seq, completion):
            total += 1
            expected_prob[0, SOS_token] += 1
            expected_prob[1:l + 1, 0] += 1 
            expected_prob[l + 1:2 * l + 1, 1] += 1 
            expected_prob[2 * l + 1, EOS_token] += 1
            
            if 2 * l + 2 < seq.shape[0]:
                expected_prob[2 * l + 2:, PAD_token] += 1
            
    if total == 0:
        return (None, 'Undefined error')
    
    expected_prob[:][:] /= total
    return ('expected_prob', expected_prob)

# ------------------- TESTING  -------------------

def verify_correct_determineTokenDistribution_sanity(max_n=6):
    """
    correct_determineTokenDistribution must find exactly 1 completion for any fully-unmasked valid sequence,
    and that completion must give probability 1.0 everywhere.
    """
    failures = []
    for n in range(1, max_n + 1):
        L = 2 * n + 2
        seq = torch.tensor([SOS_token] + [0]*n + [1]*n + [EOS_token], dtype=torch.long)
        result = correct_determineTokenDistribution(seq, vocab_size=6)
        if result[0] is None:
            failures.append((seq, 'got None for fully valid sequence'))
            continue
        if not torch.all(result[1].max(dim=-1).values == 1.0):
            failures.append((seq, 'probabilities not deterministic for unmasked sequence'))
    
    if not failures:
        print(f'correct_solve sanity check passed for n=1..{max_n}')
    else:
        print('SANITY FAILURES:', failures)


def gen_random_test(vocab_size):
    n = 1 + random.randint(5, 20)
    seq = torch.zeros((n,))
    seq[0] = SOS_token
    
    probs = [0] * vocab_size
    probs[0] = probs[1] = 0.2
    probs[SOS_token] = 0.0
    probs[EOS_token] = 0.05
    probs[PAD_token] = 0.05
    probs[MASK_token] = 0.5
    
    seq[1:] = torch.tensor(np.random.choice(vocab_size, size=(n-1,), p=probs)).to(torch.long)
    
    return seq

def exhaustive_tests(max_seq_len=14):
    """
    Enumerate every valid completion up to max_seq_len,
    then every possible subset of positions to mask.
    Yields all resulting (partially masked) sequences.
    """
    tests = set()
    
    for L in range(4, max_seq_len + 1):
        # all valid n for this length
        for n in range(1, (L - 2) // 2 + 1):
            if 2 * n + 2 > L:
                continue
            pad_len = L - (2 * n + 2)
            completion = (
                [SOS_token] + [0]*n + [1]*n + [EOS_token] + [PAD_token]*pad_len
            )
            # every subset of maskable positions (skip position 0: always SOS)
            maskable = list(range(1, L))
            for mask_bits in product([False, True], repeat=len(maskable)):
                seq = list(completion)
                for pos, do_mask in zip(maskable, mask_bits):
                    if do_mask:
                        seq[pos] = MASK_token
                tests.add(tuple(seq))
    
    return [torch.tensor(list(t), dtype=torch.long) for t in tests]

def exhaustive_single_long(n=10):
    """All 2^(2n+1) masking patterns for one long sequence."""
    L = 2 * n + 2
    completion = [SOS_token] + [0]*n + [1]*n + [EOS_token]
    tests = set()
    for mask_bits in product([False, True], repeat=L-1):
        seq = list(completion)
        for pos, do_mask in zip(range(1, L), mask_bits):
            if do_mask:
                seq[pos] = MASK_token
        tests.add(tuple(seq))
    return [torch.tensor(list(t), dtype=torch.long) for t in tests]

def boundary_stress_tests():
    """
    Hand-crafted cases targeting boundary arithmetic in determineTokenDistribution().
    Each comment explains what boundary condition it stresses.
    """
    M = MASK_token
    S, E, P = SOS_token, EOS_token, PAD_token
    
    return [
        # --- lower_bound driven by zero_last ---
        torch.tensor([S, 0, M, M, M, M, M, M], dtype=torch.long),       # zero at pos 1, forces lb>=1
        torch.tensor([S, 0, 0, M, M, M, M, M], dtype=torch.long),       # zero at pos 2, forces lb>=2
        torch.tensor([S, M, 0, M, M, M, M, M], dtype=torch.long),       # zero at pos 2 via mask
        torch.tensor([S, M, M, 0, M, M, M, M, M, M, P], dtype=torch.long), # late zero, tight lb
        
        # --- lower_bound driven by one_last ---
        torch.tensor([S, M, M, M, 1, M, M, M], dtype=torch.long),       # one at pos 4, lb >= ceil(5/2)=3
        torch.tensor([S, M, M, M, M, M, 1, M, P], dtype=torch.long),    # one at pos 6, lb >= 3
        torch.tensor([S, M, M, M, M, M, M, 1, P], dtype=torch.long),    # one at pos 7, lb >= 4
        
        # --- upper_bound driven by one_first ---
        torch.tensor([S, M, 1, M, M, M, M, M], dtype=torch.long),       # one at pos 2, ub <= 1
        torch.tensor([S, M, M, 1, M, M, M, M], dtype=torch.long),       # one at pos 3, ub <= 2
        
        # --- upper_bound driven by pad_first ---
        torch.tensor([S, M, M, M, M, M, P, P], dtype=torch.long),       # pad at 5, ub <= 1
        torch.tensor([S, M, M, M, M, P, P, P], dtype=torch.long),       # pad at 4, ub <= 1
        
        # --- lb == ub (single valid completion, deterministic) ---
        torch.tensor([S, 0, M, 1, M, E, P, P], dtype=torch.long),
        torch.tensor([S, M, M, M, M, M, M, E], dtype=torch.long),       # EOS fixes n=3
        torch.tensor([S, 0, 0, 0, M, M, M, E], dtype=torch.long),       # EOS + zeros given
        
        # --- EOS present, odd/even boundary ---
        torch.tensor([S, M, M, E, P, P, P, P], dtype=torch.long),       # EOS at pos 3, n=1
        torch.tensor([S, M, M, M, M, E, P, P], dtype=torch.long),       # EOS at pos 5, n=2
        torch.tensor([S, M, M, M, M, M, M, E], dtype=torch.long),       # EOS at pos 7, n=3
        
        # --- EOS at even position (should be invalid) ---
        torch.tensor([S, M, E, P, P, P, P, P], dtype=torch.long),       # EOS at pos 2 = invalid
        torch.tensor([S, M, M, M, E, P, P, P], dtype=torch.long),       # EOS at pos 4 = invalid
        
        # --- contradictory constraints (should be None) ---
        torch.tensor([S, 1, M, M, M, M, M, M], dtype=torch.long),       # 1 at pos 1
        torch.tensor([S, M, M, M, 0, M, M, M], dtype=torch.long),       # 0 after 1 forced
        torch.tensor([S, M, 1, 0, M, M, M, M], dtype=torch.long),       # 1 before 0
        
        # --- maximum ambiguity (all masked) ---
        torch.tensor([S] + 7*[M], dtype=torch.long),
        torch.tensor([S] + 9*[M], dtype=torch.long),
        torch.tensor([S] + 11*[M], dtype=torch.long),
        torch.tensor([S] + 5*[M] + [P], dtype=torch.long),
        torch.tensor([S] + 7*[M] + [P], dtype=torch.long),
    ]
    
def gen_structured_random_test(max_n=8):
    """
    Generate a random valid sequence, then mask a random subset of positions.
    Much more likely to produce structurally interesting partial sequences.
    """
    n = random.randint(1, max_n)
    L = random.randint(2*n + 2, 2*n + 2 + random.randint(0, 4))  # allow some padding
    completion = [SOS_token] + [0]*n + [1]*n + [EOS_token] + [PAD_token]*(L - 2*n - 2)
    
    seq = list(completion)
    # mask each non-SOS position independently, bias toward masking more
    mask_prob = random.uniform(0.3, 0.9)
    for i in range(1, L):
        if random.random() < mask_prob:
            seq[i] = MASK_token
    
    return torch.tensor(seq, dtype=torch.long)

def testing(vocab_size, tests, n_random_tests=0):
    failed_tests = []
    
    for test in tqdm(tests, 'Running custom tests'):
        obtained, expected = determineTokenDistribution(test, vocab_size), correct_determineTokenDistribution(test, vocab_size)
        
        if obtained[0] is None or expected[0] is None:
            if obtained[0] != expected[0]:
                failed_tests.append((test, obtained, expected))
            continue
        
        if not torch.allclose(obtained[1], expected[1], atol=1e-6):
            failed_tests.append((test, obtained, expected))
    
    random_tests = []
    for n_test in tqdm(range(n_random_tests), 'Running random tests'):
        test = gen_random_test(vocab_size)
        random_tests.append(test)
        
        obtained, expected = determineTokenDistribution(test, vocab_size), correct_determineTokenDistribution(test, vocab_size)
        
        if obtained[0] is None or expected[0] is None:
            if obtained[0] != expected[0]:
                failed_tests.append((test, obtained, expected))
            continue
        
        if not torch.allclose(obtained[1], expected[1], atol=1e-6):
            failed_tests.append((test, obtained, expected))
            
    total = (len(tests) + n_random_tests)
    
    if not failed_tests:
        print(f'All tests passed (total: {total})')
        return
    
    print(f'{len(failed_tests)} test(s) failed of {total} total.')
    print('Failed tests:')
    for test, obtained, expected in failed_tests:
        print('Test:', test)
        print('Obtained:', obtained)
        print('Expected:', expected)
    print(f'{len(failed_tests)} test(s) failed of {total} total.')

    return

def main():
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # model = oracleModel(vocab_size=6, device='cpu')
    # unmasker = ScheduledUnmasker(model, device='cpu', T=100, denoise="0", oracle=True)
    
    # start_seq = torch.tensor([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    # seq = torch.tensor([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 5, 5, 5, 1, 1, 1, 5, 1, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 5, 5, 4, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 5, 4, 5, 4, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 4, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 4, 4, 5, 5, 4, 5, 4, 5, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 4, 5, 4, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 4])
    # next_seq = torch.tensor([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 5, 5, 5, 1, 1, 1, 5, 1, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 5, 5, 4, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 5, 4, 5, 4, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 4, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 4, 4, 5, 5, 4, 5, 4, 5, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 5, 4, 4, 5, 4, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 4, 2, 4, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 4])
    
    # final, steps, timesteps = unmasker(start_seq, ((start_seq == MASK_token).sum() / torch.numel(start_seq)), 
    #                                    strategy='none', 
    #                                    temperature=0.1, 
    #                                    return_steps=True)
    
    # print(final)
    
    # print((next_seq==2).nonzero(as_tuple=True)[0])
    # res = determineTokenDistribution(seq, vocab_size=6, device='cpu')[1]
    
    # print(res[:, 2].nonzero(as_tuple=True)[0])
    
    exit(0)
    
    tests_no_EOS = [torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, 1, MASK_token, MASK_token, MASK_token, MASK_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, 1, 1, MASK_token, MASK_token, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token] + 7 * [MASK_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, 0, MASK_token, MASK_token, 1, MASK_token, MASK_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, 0, MASK_token, 1, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, 1, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, 1, MASK_token, 1, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, 0, 0, MASK_token, MASK_token, MASK_token, 1, MASK_token, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, 0, MASK_token, MASK_token, 0, MASK_token, MASK_token, 1, MASK_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, 1, PAD_token], dtype=torch.long)]
    
    tests_with_EOS = [torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, 1, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, 0, 0, 1, 1, MASK_token, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token] + 7 * [MASK_token] + [EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, 0, MASK_token, MASK_token, 1, MASK_token, MASK_token, EOS_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, 1, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token, PAD_token], dtype=torch.long),
             torch.tensor([SOS_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, MASK_token, EOS_token, MASK_token, PAD_token], dtype=torch.long)]
    
    print('Verifying correct_determineTokenDistribution sanity...')
    verify_correct_determineTokenDistribution_sanity(max_n=200)
    
    print('Testing on single long sequence with all masking patterns')
    single_long_tests = exhaustive_single_long(n=10)
    testing(vocab_size=6, tests=single_long_tests, n_random_tests=0)
    
    print('Testing on exhaustive tests on length up to 16')
    wide_tests = exhaustive_tests(max_seq_len=16)
    testing(vocab_size=6, tests=wide_tests, n_random_tests=0)

    print('Testing on boundary stress tests')
    boundary_tests = boundary_stress_tests()
    testing(vocab_size=6, tests=boundary_tests, n_random_tests=0)
    
    print('Testing on structured random tests')
    structured_random_tests = [gen_structured_random_test(max_n=max_n) for _ in range(500) for max_n in range(1, 100)]
    testing(vocab_size=6, tests=structured_random_tests, n_random_tests=0)
    
    print('Testing on completely random tests')
    testing(vocab_size=6, tests=[], n_random_tests=10000)
    
    print('Testing on custom edge cases')
    testing(vocab_size=6, tests=tests_no_EOS + tests_with_EOS, n_random_tests=0)
    
    print('All testing done.')
    
    exit(0)
    
if __name__ == '__main__':
    main()
    
'''
Testing log:

Testing on single long sequence with all masking patterns
Running custom tests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2097152/2097152 [33:28<00:00, 1044.07it/s]
Running random tests: 0it [00:00, ?it/s]
All tests passed (total: 2097152)
Testing on exhaustive tests on length up to 16
Running custom tests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 393218/393218 [04:32<00:00, 1445.61it/s]
Running random tests: 0it [00:00, ?it/s]
All tests passed (total: 393218)
Testing on boundary stress tests
Running custom tests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:00<00:00, 1834.24it/s]
Running random tests: 0it [00:00, ?it/s]
All tests passed (total: 27)
Testing on structured random tests
Running custom tests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 49500/49500 [04:35<00:00, 179.38it/s]
Running random tests: 0it [00:00, ?it/s]
All tests passed (total: 49500)
Testing on completely random tests
Running custom tests: 0it [00:00, ?it/s]
Running random tests: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:04<00:00, 2417.78it/s]
All tests passed (total: 10000)
Testing on custom edge cases
Running custom tests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:00<00:00, 2444.73it/s]
Running random tests: 0it [00:00, ?it/s]
All tests passed (total: 23)
All testing done.

'''