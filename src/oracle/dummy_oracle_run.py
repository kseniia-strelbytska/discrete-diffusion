import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from oracle.grammar_oracles import oracleModel
from datasets.constants import *

oracle = oracleModel(grammar_name='anbn', vocab_size=6, device='cpu')
input = torch.tensor([SOS_token] + [MASK_token] * 20, dtype=torch.long).unsqueeze(0)
output = oracle(input)

greedy_select = output.argmax(dim=-1)

print(output)
print("Greedy selection:", greedy_select)
