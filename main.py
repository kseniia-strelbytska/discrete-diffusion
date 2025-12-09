import train
from data_generation import Dataset, sample_uniform_t, sample_inverse_t, sample_masked, generate_seq, satisfies_rule_2, select_satisfies_rule_2
from train import train_model, inference
from unmask import get_unmasker
from model import Model, TransformerClassifier
from loss import rblb
from torch.optim import Adam, AdamW
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from unmask import Unmasker, get_unmasker
from noise_schedule_unmask import ScheduledUnmasker, SequencedScheduledUnmasker
import matplotlib.pyplot as plt

def gather_stats(model):
    exact, valid, invalid = [], [], []

    changed_tokens, total_tokens = 0, 0

    for X, y, timestep in tqdm(ds.data):
        # remove batch

        y_pred = model(X, timestep)[0].argmax(0)

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

def gather_training_stats(train_dataloader):
    n_ones, n_masks, samples = 0, 0, 0

    for X, y, alpha in tqdm(train_dataloader, desc="Processing training samples"):
        y = torch.argmax(y, dim=1)

        masks = y[(X==2)]

        n_ones += torch.sum(masks)
        n_masks += torch.numel(masks)
        samples += X.shape[0]

    print(f'Total number of masks {n_masks}/{samples * model.l}')
    print(f'Average number of masks per training sample {n_masks / samples}/{model.l}')
    print(f'Number of masks unmasked as 1s {n_ones} ({n_ones/n_masks})')


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
num_workers = 30 if device == torch.device("cuda") else 0

seq_len = 20

model = TransformerClassifier(device=device, vocab_size=3, num_layers=7, embedding_size=128, l=seq_len).to(device)
ds = Dataset(seq_len, 1.0, 10**8, False)
print("Generated Dataset")

train_dataloader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

loss = rblb(device).to(device)
optim = AdamW(model.parameters(), 0.01)

unmask_model = SequencedScheduledUnmasker(model, 0.02)

seqs = generate_seq(model.l)
data = sample_masked(model.l, 100, torch.full((10**5, ), torch.tensor(0.5)), seqs)[:, 0, :] # (batch, 2, l) -> (batch, l)

# print(data[0])
# f = unmask_model(data[0].unsqueeze(0).to(device), torch.full((1, ), torch.tensor(0.5)).to(device))
# print(f)

model = train_model(model=model, data_loader=train_dataloader, loss_fn=loss, optimizer=optim, device=device, num_epochs=50000, dict_path='models/test/', figure_path='figures/test/')

exit(0)

# model = train_model(model, train_dataloader, loss, optim, device, 50)

# To do:
# one test 
# lr scheduling

# torch.save(model.state_dict(), './models/diffusion_model_31_10')

# device = "cpu"

np.random.seed(1)

# model = Model(dim=20, category_count=2, hidden_count1=256, hidden_count2=256)

# model = train_model(model, train_dataloader, loss, optim, device, 50, './recentmodels/')

# exit(0)

model.load_state_dict(torch.load('./models/diffusion_model_31_10_80epochs'))
print("Loaded model (pre-trained for 80 epochs)")

unmasker = get_unmasker(model)
correct = 0
cs = []

for X, y in tqdm(ds.data):
    x_pred = unmasker(X.unsqueeze(0))

    cnt_1 = (x_pred == 1).sum(1)
    if cnt_1 == 10:
        correct += 1 

    cs.append(cnt_1[0].item())

correct = sum([1 if i == 10 else 0 for i in cs])

print(correct / len(cs))

plt.hist(cs, bins=seq_len, range = [0, seq_len])
plt.xticks(range(0, seq_len + 1))
plt.savefig(f'./figures/scheduled_unmasker_all')

exit(0)

