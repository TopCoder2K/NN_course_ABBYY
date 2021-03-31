import torch
import numpy as np

from nn.base import Module


class FullyConnectedLayer(Module):
    """
    Слой, осуществляющий линейное преобразоваие: Y = X @ W + b, Y \in R^{N x n_out}, X \in R^{N, n_in}.

    Аттрибуты
    ---------
    `W` : torch.tensor, shape=(n_in, n_out)
        Матрица размера (n_in, n_out), где в данном случае n_in равно числу признаков,
        а n_out равно количеству нейронов в слое.
    `b` : torch.tensor, shape=(N, n_out)
        Вектор свободных членов, по одному числу на один нейрон.
    `gradW` : torch.tensor, shape=(n_in, n_out)
        Хранит градиент матрицы весов линейного слоя.
    `gradb` : torch_tensor, shape=(N, n_out)
        Хранит градиент вектора свободных членов.
    """

    def __init__(self, in_features, out_features, bias=True, init=None, optimizer=None):
        """
        Параметры
        ---------
        in_features : integer
            Число фич у входа слоя.
        out_features : integer
            Число фич у выхода слоя.
        bias : boolean
            Нужен ли bias?
        init : ?????????????????????
        optimizer : ???????????????????
        """
        super(FullyConnectedLayer, self).__init__()

        if init is not None:
            self.W = init
        else:
            self.W = torch.full((out_features, in_features), fill_value=0.)
        if bias:
            self.b = torch.full((list(self.W.shape)[-1], 1), fill_value=0.)
        else:
            self.b = None
        self.gradW = None
        self.gradb = None

    def forward(self, x):
        self.output = torch.dot(self.W, x)
        if self.b is not None:
            self.output += self.b

        return self.output

    def backward(self, x, grad_output):
        raise NotImplementedError  # TODO: replace line with your code

    def zero_grad(self):
        self.gradW.fill(0.)
        self.gradb.fill(0.)

    def update_grad_input(self, module_input, grad_output):
        self.grad_input = grad_output @ self.W
        return self.grad_input

    def update_grad_params(self, module_input, grad_output):
        self.gradW = torch.dot(module_input.T, grad_output)
        self.gradb = torch.sum(grad_output, dim=0)

        assert self.gradb.shape == self.b.shape

    @property
    def parameters(self):
        return [self.W, self.b]

    @property
    def grad_params(self):
        return [self.gradW, self.gradb]

    # def apply_grad(self):
    #     raise NotImplementedError  # TODO: replace line with your code
