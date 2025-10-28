import torch

class Model(torch.nn.Module):
    def __init__(self, dim, category_count, hidden_count1, hidden_count2):
        super().__init__()
        self.dim = dim
        self.category_count = category_count
        self.l1 = torch.nn.Linear(dim, hidden_count1)
        self.l2 = torch.nn.Linear(hidden_count1, hidden_count2)
        self.l3 = torch.nn.Linear(hidden_count2, dim*category_count)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        X = self.l1(X)
        X = self.relu(X)
        X = self.l2(X)
        X = self.relu(X)
        X = self.l3(X)

        return X.view(-1, self.category_count, self.dim)

