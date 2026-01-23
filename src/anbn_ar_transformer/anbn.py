import torch 
import torch.nn as nn
import numpy as np
from formal_grammar import FormalGrammar
from constants import EOS_token, SOS_token, PAD_token, MASK_token

'''

Rule 1: the number of 0s and 1s match
Rule 2: 0s precede 1s

'''

class anbnGrammar(FormalGrammar):
    # Rule 1: the number of 0s and 1s match

    def __init__(self, l):
        super().__init__(l)
        self.data = None
        self.default_eval_type = 'next_token'

    def does_satisfy_rule1(self, seq):
        zeros = (seq == 0).sum()
        ones = (seq == 1).sum()
                
        return (zeros == ones)

    # Rule 2: 0s preceed 1s
    def does_satisfy_rule2(self, seq): # seq has no batch dim, no SOS token
        zero, one = False, False
        for idx in range(0, len(seq)):
            if seq[idx] != 0 and seq[idx] != 1: # one of EOS/SOS/PAD/MASK tokens
                continue 
                
            if seq[idx] == 0:
                zero = True 
                
                if one == True:
                    return False 
            else:
                one = True 
        return True

    def evaluate(self, seq):
        a = self.does_satisfy_rule1(seq)
        b = self.does_satisfy_rule2(seq)
        
        return np.array([int(i) for i in [a, b, (a & b)]])

    def generate_seq(self): # max length
        # returns tensor shaped (#seqs, length + 2) 
        # as SOS and EOS tokens are added
        data = None 
        
        for l in range(2, self.l + 1, 2):
            seq = torch.cat([torch.tensor([SOS_token]),
                            torch.zeros(l // 2), 
                            torch.ones(l // 2), 
                            torch.tensor([EOS_token]),
                            torch.full((self.l - l,), torch.tensor(PAD_token))], dim=-1).long().unsqueeze(0)
            
            if data == None:
                data = seq.clone()
            else:
                data = torch.cat([data, seq], dim=0)
        
        print(f'Data generated; shape: {data.shape}')

        return data
