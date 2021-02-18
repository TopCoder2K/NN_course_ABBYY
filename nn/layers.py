import torch
from nn.base import Module


class FullyConnectedLayer(Module):
    def __init__(self, in_features, out_features, bias=True, init=None, optimizer=None):
        super(FullyConnectedLayer, self).__init__()
        raise NotImplementedError  # TODO: replace line with your code

    def forward(self, x):
        raise NotImplementedError  # TODO: replace line with your code

    def backward(self, x, grad_output):
        raise NotImplementedError  # TODO: replace line with your code

    def zero_grad(self):
        raise NotImplementedError  # TODO: replace line with your code

    def apply_grad(self):
        raise NotImplementedError  # TODO: replace line with your code
