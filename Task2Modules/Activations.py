import torch


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return torch.maximum(X, torch.tensor(0.))

    def backward(self, new_deltaL):
        return torch.mul(new_deltaL, self.input > 0.)
