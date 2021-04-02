import torch
import numpy as np

from nn.base import Module


class FullyConnectedLayer(Module):
    """
    Слой, осуществляющий линейное преобразоваие: Y = X @ W + b, Y \in R^{N x n_out}, X \in R^{N, n_in}.

    Атрибуты
    --------
    `W` : torch.tensor, shape=(in_features, out_features)
        Матрица размера (in_features, out_features), где в данном случае in_features равно числу признаков,
        а out_features равно количеству нейронов в слое.
    `b` : torch.tensor, shape=(N, out_features)
        Вектор свободных членов, по одному числу на один нейрон.
    `gradW` : torch.tensor, shape=(in_features, out_features)
        Хранит градиент матрицы весов линейного слоя.
    `gradb` : torch_tensor, shape=(N, out_features)
        Хранит градиент вектора свободных членов.
    """

    def __init__(self, in_features, out_features, bias=True, init=None, optimizer=None):
        """
        Параметры
        ---------
        `in_features` : integer
            Число фич у входа слоя.
        `out_features` : integer
            Число фич у выхода слоя.
        `bias` : boolean
            Нужен ли bias?
        `init` : torch.tensor с shape=(in_features, out_features) или array like из тензоров с shape=(1, 2)
            Содержит значения, которыми иницализируются параметры слоя. Если ```bias == True```, то содержит в себе
            два тензора: начальные значения для W и b соответственно.
        `optimizer` : ??????????????????? TODO
        """
        super(FullyConnectedLayer, self).__init__()

        if init is not None:
            if bias:
                self.W, self.b = init
            else:
                self.W = init
        else:
            stdv = 1. / np.sqrt(in_features)
            self.W = torch.tensor(np.random.uniform(-stdv, stdv, size=(in_features, out_features)))
            if bias:
                self.b = torch.tensor(np.random.uniform(-stdv, stdv, size=out_features))
            else:
                self.b = None

        self.gradW = torch.full((out_features, in_features), fill_value=0.)
        self.gradb = torch.full((list(self.W.shape)[-1], 1), fill_value=0.)

    def forward(self, module_input):
        self.output = torch.dot(self.W, module_input)
        if self.b is not None:
            self.output += self.b

        return self.output

    def zero_grad(self):
        self.gradW = torch.zeros(self.gradW.shape)  # TODO: есть ли способ получше?
        self.gradb = torch.zeros(self.gradb.shape)

    def update_module_input_grad(self, module_input, grad_output):
        self.grad_input = grad_output @ self.W.T
        return self.grad_input

    def update_params_grad(self, module_input, grad_output):
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
    #     raise NotImplementedError


class Softmax(Module):
    """Осуществляет softmax-преобразование. Подробности по формулам см. в README.md."""

    def forward(self, module_input):
        # Нормализуем для численной устойчивости
        self.output = np.exp(module_input - module_input.max(dim=1, keepdims=True))
        self.output /= self.output.sum(dim=1).reshape(-1, 1)

        return self.output

    # TODO: попробовать сделать без циклов, но это непросто.........
    def update_module_input_grad(self, module_input, grad_output):
        for i in range(self.output.shape[0]):
            softmax_i = self.output[i, :]
            partial_softmax = -torch.matmul(softmax_i.T, softmax_i) + torch.diag(softmax_i)
            for j in range(self.output.shape[1]):
                self.grad_input[i, j] = torch.mul(grad_output[i, :], partial_softmax[:, j])

        return self.grad_input

