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
        The seq supplied is truncated at the first EOS token (if it exists) for evaluation.
        Thus the desired format for the strings is 
        SOS [0/1] EOS
        
        '''
        
        if (seq == MASK_token).long().sum() != 0: # contains masked tokens
            return False 
        
        SOS_token_count, EOS_token_count = (seq == SOS_token).long().sum(), (seq == EOS_token).long().sum()
        if SOS_token_count != 1 or EOS_token_count != 1:
            # more than one SOS/EOS token, or missing SOS/EOS token
            return False
        
        if seq[0] != SOS_token or seq[-1] != EOS_token:
            # SOS token is not at the start, or EOS token is not at the end
            return False
        
        zero_count, one_count = (seq[1:-1] == 0).long().sum(), (seq[1:-1] == 1).long().sum()
        if zero_count + one_count != len(seq) - 2:
            # some other token (i.e. PAD) is present between SOS and EOS
            return False
            
        return True
        
    def evaluate(self, seq):
        # Ignore tokens after the first EOS token for evaluation.
        # If EOS is absent, keep the current behavior and use the full sequence.
        eos_positions = torch.where(seq == EOS_token)[0]
        eval_seq = seq[:eos_positions[0] + 1] if len(eos_positions) > 0 else seq

        a = self.does_satisfy_rule1(eval_seq)
        b = self.does_satisfy_rule2(eval_seq)
        c = self.does_satisfy_format(eval_seq)
        
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

        self.data = data
