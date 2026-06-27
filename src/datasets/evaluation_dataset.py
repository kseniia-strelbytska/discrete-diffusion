"""
EvaluationDataset: fixed prompt sets for the a^n b^n grammar.

For other grammars, use eval_dataset='unconditional' which generates
fully-masked sequences compatible with any grammar.
"""

import torch
from .constants import EOS_token, SOS_token, PAD_token, MASK_token
from .anbn import anbnGrammar
from .dataset import Dataset, get_fixed_dataset


class EvaluationDataset():
    '''
    Expected init parameters:
        l: Length of strings (excluding SOS/EOS)
        eval_dataset: Type of dataset (see below)
        eval_type: Eval type is either 'full' or 'random'
        n_samples: number of samples to take if eval_type='random'

    Dataset types:
    -- limited:
        Contains l samples. For each l0 in [1, l//2], adds:
        SOS 000...0 MASK...  (l0 zeros)
        SOS 000...01 MASK... (l0 zeros, one '1')
    -- randomised:
        100 samples. For each l0 in [8,32], 4 random l1 values:
        SOS 000...011...1 MASK... (l0 zeros, l1 ones)
    -- complete:
        All OOD sequences (l0 in [32,64], l1 in [0, 64-l0])
    -- diffusion:
        Noisy samples drawn from the anbn training distribution
    -- unconditional:
        500 fully-masked sequences (grammar-agnostic)
    '''

    def __init__(self, l, eval_dataset, eval_type='full', n_samples=100,
                 T=None, sampling_eps=None, device=None):
        self.l = l
        self.eval_dataset = eval_dataset
        self.eval_type = eval_type
        self.n_samples = n_samples
        self.T = T
        self.sampling_eps = sampling_eps
        self.device = device

        self.full_data = []
        if eval_dataset == 'limited':
            self._init_limited()
        elif eval_dataset == 'randomised':
            self._init_randomised()
        elif eval_dataset == 'complete':
            self._init_complete()
        elif eval_dataset == 'diffusion':
            self._init_diffusion()
        elif eval_dataset == 'unconditional':
            self._init_unconditional()

        self.sampled_data = self.full_data.clone()[torch.randperm(self.full_data.shape[0])][:n_samples]
        self.data = self.full_data.clone() if eval_type == 'full' else self.sampled_data

    def _init_limited(self):
        for l0 in range(1, self.l // 2 + 1):
            self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [MASK_token]*(self.l + 1 - l0)).unsqueeze(0))
            self.full_data.append(torch.tensor([SOS_token] + [0]*l0 + [1] + [MASK_token]*(self.l - l0)).unsqueeze(0))
        self.full_data = torch.cat(self.full_data, dim=0)

    def _init_randomised(self):
        for l0 in range(8, 33):
            sampled_l1 = torch.randperm(l0)[:4] + 1
            for l1 in sampled_l1:
                self.full_data.append(
                    torch.tensor([SOS_token] + [0]*l0 + [1]*l1 + [MASK_token] * (self.l + 1 - l0 - l1)).unsqueeze(0)
                )
        self.full_data = torch.cat(self.full_data, dim=0)

    def _init_complete(self):
        for l0 in range(32, 65):
            for l1 in range(0, 64 - l0 + 1):
                self.full_data.append(
                    torch.tensor([SOS_token] + [0]*l0 + [1]*l1 + [MASK_token] * (self.l + 1 - l0 - l1)).unsqueeze(0)
                )
        self.full_data = torch.cat(self.full_data, dim=0)

    def _init_diffusion(self):
        grammar = anbnGrammar(self.l)
        grammar.generate_seq()
        grammar.data = grammar.data[torch.randperm(grammar.data.shape[0])]
        dataset = Dataset(grammar.data, self.device, self.T, self.sampling_eps)
        fixed_dataset = get_fixed_dataset(dataset, self.device, batch_size=self.l // 2)
        self.full_data = fixed_dataset[0][0]

    def _init_unconditional(self):
        self.full_data = torch.concat(
            [torch.full((500, 1), SOS_token).long(),
             torch.full((500, self.l + 1), MASK_token).long()],
            dim=1
        )
