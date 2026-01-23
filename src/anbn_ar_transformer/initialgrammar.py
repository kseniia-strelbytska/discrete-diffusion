import torch 
import torch.nn as nn
import itertools
import numpy as np
from formal_grammar import FormalGrammar
from constants import EOS_token, SOS_token, PAD_token, MASK_token

'''

Rule 1: the number of 0s and 1s match
Rule 2: string does not contain "010" or "101" as substring

'''

class initialGrammar(FormalGrammar):
    def __init__(self, l):
        super().__init__(l)
        self.data = None
        self.default_eval_type = 'prefix'
    
    # Rule 1: the number of 0s and 1s match
    def does_satisfy_rule1(self, seq):
        zeros = (seq == 0).sum()
        ones = (seq == 1).sum()
                
        return (zeros == ones)

    # Rule 2: string does not contain "010" or "101" as substring
    def does_satisfy_rule2(self, seq):
        for idx in range(1, len(seq) - 1):
            if seq[idx] != seq[idx - 1] and seq[idx] != seq[idx + 1]:
                return False 
        return True
    
    def evaluate(self, seq):
        a = self.does_satisfy_rule1(seq)
        b = self.does_satisfy_rule2(seq)
        
        return np.array([int(i) for i in [a, b, (a & b)]])

    def generate_seq(self):
        # generate all of length l that satisfiy rule 1
        idxs = torch.arange(self.l).tolist()
        combs = list(itertools.combinations(idxs, self.l // 2))
        data = torch.zeros(len(combs), self.l).long()
        data[torch.arange(len(combs))[:, None], combs] = 1
        
        # select those that satisfy rule 2
        valid = []
        for seq in data:
            valid_seq = True 
            for idx in range(1, len(seq) - 1):
                if seq[idx] != seq[idx - 1] and seq[idx] != seq[idx + 1]:
                    valid_seq = False 
                    
            if valid_seq == True:
                valid.append(seq.unsqueeze(0))

        valid = torch.cat(valid, dim=0)
        valid = torch.cat([torch.full((valid.shape[0], 1), SOS_token), 
                           valid, 
                           torch.full((valid.shape[0], 1), EOS_token)], dim=-1).long()
        
        ## TO CHECK SHAPE

        return valid
