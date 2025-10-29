import train
from data_generation import Dataset
from train import train_model
from model import Model
from loss import rblb
from torch.optim import Adam
import torch

ds = Dataset(20, 0.01)
train_dataloader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
loss = rblb()
model = Model(20, 2, 256, 256)
optim = Adam(model.parameters(), 0.01)
model = train_model(model, train_dataloader, loss, optim, 30)