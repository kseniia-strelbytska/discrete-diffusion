import train
from data_generation import Dataset, sample_uniform_t, sample_inverse_t, sample_masked
from train import train_model
from model import Model, TransformerClassifier
from loss import rblb
from torch.optim import Adam, AdamW
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(1)

# t1 = sample_uniform_t(10)
# t2 = sample_inverse_t(10)

# seqs = sample_masked(20, 10, t2)

# print(seqs)

# # print(t1[0:10], '\n', t2[0:10])

# exit(0)

ds = Dataset(20, 0.01)

train_dataloader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
loss = rblb().to(device)

# model = Model(dim=20, category_count=2, hidden_count1=256, hidden_count2=256)

model = TransformerClassifier(vocab_size=3, num_layers=2, embedding_size=8, l=20).to(device)

optim = AdamW(model.parameters(), 0.01)
model = train_model(model, train_dataloader, loss, optim, device, 30)

# one test 
# lr scheduling

torch.save(model.state_dict(), './diffusion_model')