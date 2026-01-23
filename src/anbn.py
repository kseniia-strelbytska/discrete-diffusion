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
    
    # checks for format
    def does_satisfy_format(self, seq):
        '''
        The format for the strings is 
        SOS 000...0111...1 EOS PAD
        '''
        
        if (seq == MASK_token).long().sum() != 0: # contains masked tokens
            return False 
        
        ordered_tokens = [SOS_token, 0, 1, EOS_token, PAD_token] # the correct order of tokens
        available = [1, 10**9, 10**9, 1, 10**9] # max number of each token
        token_pos = 0
        
        for idx in range(len(seq)):
            # use a loop (it is allowed for some tokens to be missing completely, e.g. SOS 0 EOS is valid)
            while token_pos < len(ordered_tokens) and seq[idx] != ordered_tokens[token_pos]:
                token_pos += 1 
            
            if token_pos >= len(ordered_tokens) or seq[idx] != ordered_tokens[token_pos]:
                # either token not in the allowed set / token seen before is met
                return False 
            available[token_pos] -= 1
            if available[token_pos] < 0:
                # more than one SOS/EOS
                return False
            
        if available[0] != 0 or available[3] != 0:
            # SOS/EOS is missing
            return False 
            
        return True
        
    def evaluate(self, seq):
        a = self.does_satisfy_rule1(seq)
        b = self.does_satisfy_rule2(seq)
        c = self.does_satisfy_format(seq)
        
        return np.array([int(i) for i in [a, b, (a & b), c]])

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
