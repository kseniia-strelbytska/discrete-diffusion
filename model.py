import torch

class Model(torch.nn.Module):
    def __init__(self, d, cat, dhid1, dhid2):
        super().__init__()
        self.l1 = torch.nn.Linear(d, dhid1)
        self.l2 = torch.nn.Linear(dhid1, dhid2)
        self.l3 = torch.nn.Linear(dhid2, d*cat)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        X = self.l1(X)
        X = self.relu(X)
        X = self.l2(X)
        X = self.relu(X)
        X = self.l3(X)

        return X
    
model = Model(5, 10, 10)
x = torch.Tensor([2, 2, 1, 3, 2])
print(model(x))