import train
from data_generation import Dataset
from train import train_model
from model import Model, TransformerClassifier
from loss import rblb
from torch.optim import Adam
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = Dataset(20, 0.01)
train_dataloader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
loss = rblb().to(device)

model = Model(20, 2, 256, 256)

# model = TransformerClassifier(3, 2, 8, 20).to(device)
optim = Adam(model.parameters(), 0.01)
model = train_model(model, train_dataloader, loss, optim, device, 30)

# one test 
# lr scheduling

print(result)

torch.save(model.state_dict(), './diffusion_model')