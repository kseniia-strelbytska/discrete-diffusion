import train
from data_generation import Dataset, sample_uniform_t, sample_inverse_t, sample_masked
from train import train_model
from model import Model, TransformerClassifier
from loss import rblb
from torch.optim import Adam, AdamW
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
optim = AdamW(model.parameters(), 0.001)

model.load_state_dict(torch.load('./models/diffusion_model_31_10_80epochs'))

print("Loaded model (pre-trained for 80 epochs)")

def gather_stats(model):
    exact, valid, invalid = [], [], []

    changed_tokens, total_tokens = 0, 0

    for X, y in tqdm(ds.data):
        # remove batch

        y_pred = model(X)[0].argmax(0)

        if torch.equal(y_pred, y):
            exact.append((X, y_pred))
        elif y_pred.sum() == y_pred.size(0) // 2:
            valid.append((X, y_pred))

            changed_tokens += torch.where(y != y_pred, torch.ones_like(y), torch.zeros_like(y)).sum() - (y == 2).sum()
            total_tokens += (y != 2).sum()
        else:
            invalid.append((X, y_pred))

    with open('./figures/prediction_results.txt', 'w') as f:
        print(f'Changed tokens: {changed_tokens}/{total_tokens} ({changed_tokens/total_tokens:.4f}) (unmasked -> unmasked)')

        for case in [(exact, 'Exact'), (valid, 'Valid'), (invalid, 'Invalid')]:
            f.write(f"{case[1]} solutions: {len(case[0])}" + "\n")
            for X, y in case[0]:
                X = ''.join([str(i) for i in X.numpy()])
                y = ''.join([str(i) for i in y.numpy()])

                f.write(X + " " + y + "\n")

    print(f'Exact solutions: {len(exact)}, valid solutions: {len(valid)}, invalid solutions: {len(invalid)}')

# model = train_model(model, train_dataloader, loss, optim, device, 50)

# To do:
# one test 
# lr scheduling

# torch.save(model.state_dict(), './models/diffusion_model_31_10')