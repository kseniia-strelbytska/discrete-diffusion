import torch

from deterministic_token_distribution import oracleModel
from constants import *

oracle = oracleModel(vocab_size=6, device='cpu')
input = torch.tensor([SOS_token] + [MASK_token] * 20, dtype=torch.long).unsqueeze(0)
output = oracle(input)

greedy_select = output.argmax(dim=-1)

print(output)
print("Greedy selection:", greedy_select)
