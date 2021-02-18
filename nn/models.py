import numpy as np, pandas as pd, matplotlib.pyplot as plt

from nn.base import Model


class FeedForwardModel(Model):
    def __init__(self, layers=None, loss=None, optimizer=None):
        super(FeedForwardModel, self).__init__()
        raise NotImplementedError  # TODO: replace line with your implementation

    def forward(self, x):
        raise NotImplementedError  # TODO: replace line with your implementation

    def backward(self, x, y):
        raise NotImplementedError  # TODO: replace line with your implementation

    def zero_grad(self):
        raise NotImplementedError  # TODO: replace line with your implementation

    def apply_grad(self):
        raise NotImplementedError  # TODO: replace line with your implementation

    def train(self, data, n_epochs):
        raise NotImplementedError  # TODO: replace line with your implementation
