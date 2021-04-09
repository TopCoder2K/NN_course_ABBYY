import torch
import numpy as np

from nn.base import Module


class FullyConnectedLayer(Module):
    """
    Слой, осуществляющий линейное преобразование: Y = X @ W + b, Y \in R^{N x n_out}, X \in R^{N, n_in}.

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

    def __init__(self, in_features, out_features, bias=True, init=None):
        """
        Параметры
        ---------
        `in_features` : integer
            Число фич у входа слоя.
        `out_features` : integer
            Число фич у выхода слоя.
        `bias` : boolean
            Нужен ли bias?
        `init` : torch.tensor с shape=(in_features, out_features)
            Содержит значения, которыми инициализируются веса слоя. Заметим, что в реализации pytorch
            bias не инициализируется извне, у нас тоже не будет.
        """
        super(FullyConnectedLayer, self).__init__()
        self.b = None
        self.W = None

        if init is not None:
            self.W = init.clone().detach()
        else:
            stdv = 1. / np.sqrt(in_features)
            self.W = torch.tensor(np.random.uniform(-stdv, stdv, size=(in_features, out_features)), dtype=torch.float)
            if bias:
                self.b = torch.tensor(np.random.uniform(-stdv, stdv, size=out_features), dtype=torch.float)

        self.gradW = torch.full((out_features, in_features), fill_value=0., dtype=torch.float)
        self.gradb = torch.full((self.W.shape[-1], 1), fill_value=0., dtype=torch.float)

    def forward(self, module_input):
        self.output = torch.matmul(module_input, self.W)
        if self.b is not None:
            self.output += self.b

        return self.output

    def zero_grad(self):
        self.gradW.fill_(0.)
        self.gradb.fill_(0.)

    def update_module_input_grad(self, module_input, grad_output):
        self.grad_input = torch.matmul(grad_output, self.W.t())
        return self.grad_input

    def update_params_grad(self, module_input, grad_output):
        self.gradW = torch.matmul(module_input.t(), grad_output)
        if self.b is not None:
            self.gradb = torch.sum(grad_output, dim=0)

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
        # Нормализуем для численной устойчивости, а потом возводим в exp
        self.output = torch.exp(module_input - module_input.max(dim=1, keepdim=True).values)
        self.output /= self.output.sum(dim=1, keepdim=True)

        return self.output

    # TODO: попробовать сделать без циклов, но это непросто.........
    def update_module_input_grad(self, module_input, grad_output):
        # Нужно проиницилизировать self.grad_input, так как дальше берутся уже индексы
        if self.grad_input is None:
            self.grad_input = torch.zeros(size=self.output.shape)

        for i in range(self.output.shape[0]):
            softmax_i = self.output[i, :].unsqueeze(1)
            partial_softmax = -torch.matmul(softmax_i, softmax_i.t()) + torch.diag(softmax_i.squeeze())
            for j in range(self.output.shape[1]):
                self.grad_input[i, j] = torch.dot(grad_output[i, :], partial_softmax[:, j])

        return self.grad_input


class LogSoftmax(Module):
    """Осуществляет log(softmax)-преобразование. Подробности по формулам см. в README.md."""

    def forward(self, module_input):
        # Нормализуем для численной устойчивости
        self.output = module_input - module_input.max(axis=1, keepdim=True).values
        self.output = self.output - torch.log(torch.sum(torch.exp(self.output), dim=1, keepdim=True))

        return self.output

    def update_module_input_grad(self, module_input, grad_output):
        # Нормализуем для численной устойчивости, а потом уже возводим в exp
        exp_module_input = torch.exp(module_input - module_input.max(axis=1, keepdim=True).values)
        softmax = exp_module_input / torch.sum(exp_module_input, dim=1, keepdim=True)

        self.grad_input = grad_output.clone().detach()
        self.grad_input -= torch.mul(softmax, torch.sum(grad_output, dim=1, keepdim=True))

        return self.grad_input
