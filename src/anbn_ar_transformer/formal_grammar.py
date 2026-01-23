from abc import ABC, abstractmethod

class FormalGrammar(ABC):
    def __init__(self, l):
        self.l = l

    @abstractmethod
    def does_satisfy_rule1(self, seq):
        # Returns true if seq satisfies rule 1
        pass
    
    @abstractmethod
    def does_satisfy_rule2(self, seq):
        # Returns true if seq satisfies rule 2
        pass

    @abstractmethod
    def evaluate(self, seq):
        # Returns an np array [a, b, c] of ints where
        # a = 1 if does_satisfy_rule1(a) else 0
        # b = 1 if does_satisfy_rule2(a) else 0 
        # c = 1 if (a + b == 2) else 0
        
        pass
    
    @abstractmethod
    def generate_seq(self, num_samples):
        # Returns a tensor of valid sequences
        pass