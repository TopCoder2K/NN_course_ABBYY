from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np, pandas as pd, matplotlib.pyplot as plt


# TODO: стоит ли использовать raise NotImplementedError внутри абстрактного метода?
class Module(ABC):
    """
    Абстрактный класс модуля. Именно из модулей будет состоять нейронная сеть.

    Аттрибуты
    ---------
    `output` : torch.tensor
        Выход слоя.
    `grad_input` : torch.tensor
        Градиент по входу этого слоя. Нужен для удобства, так как при обратном распространении ошибки
        каждому слою нужен градиент по выходу этого слоя. Но выход этого слоя есть вход для следующего слоя,
        поэтому логично возвращать градиент по входу и передавать его далее.
    """
    def __init__(self):
        self.output = None
        self.grad_input = None

    @abstractmethod
    def forward(self, *args):
        """Вычисляет операцию слоя."""
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def backward(self, module_input, grad_output):
        """
        Осуществляет шаг backpropagation'а для этого слоя,
        дифференцируя функцию слоя по входу и по параметрам.

        Параметры
        ---------
        `module_input` : torch.tensor
            Вход слоя
        `grad_output` : torch.tensor
            Градиент по выходу этого слоя, пришедший от следующего слоя
        Возвращает
        ----------
        `self.grad_input` : torch.tensor
            Градиент функции слоя по входу.
        """

        self.update_layer_input_grad(module_input, grad_output)
        self.update_params_grad(module_input, grad_output)
        return self.grad_input

    # TODO: нужно ли возвращать градиент?
    @abstractmethod
    def update_layer_input_grad(self, module_input, grad_output):
        """
        Вычисляет градиент функции слоя по входу и возвращает его в виде `self.grad_input`.
        Размер (`shape`) поля `self.grad_input` всегда совпадает с размером `input`.

        Параметры
        ---------
        `module_input` : torch.tensor
            Вход слоя
        `grad_output` : torch.tensor
            Градиент по выходу этого слоя, пришедший от следующего слоя
        Возвращает
        ----------
        `self.grad_input` : torch.tensor
            Вычисленный градиент функции слоя по входу
        """
        raise NotImplementedError

    def update_params_grad(self, module_input, grad_output):
        """
        Вычисляет градиент функции слоя по параметрам (весам) этого слоя.
        Ничего не возвращает, только сохраняет значения градиентов в соответствующие поля.
        Так как не у всех слоёв есть параметры, этот метод необязательный.

        Параметры
        ---------
        `module_input` : torch.tensor
            Вход слоя
        `grad_output` : torch.tensor
            Градиент по выходу этого слоя, пришедший от следующего слоя
        """
        pass

    @abstractmethod
    def zero_grad(self):
        """Зануляет градиенты у параметров слоя (если они есть). Нужно для оптимизатора."""
        raise NotImplementedError

    @abstractmethod
    def apply_grad(self):
        """?????????????????????????"""
        raise NotImplementedError


class Optimizer(ABC):
    """
    Абстрактный класс оптимизатора.

    Аттрибуты
    ---------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора. Например, `learning_rate` и `momentum` для SGD с Momentum.
    `state` :  dict
        Словарь cо состоянием оптимизатора. Нужен, чтобы сохранять старые значения градиентов.
    """
    def __init__(self, *args):
        self.config = defaultdict(np.float64)
        self.state = defaultdict(np.float64)

    @abstractmethod
    def step(self, weights, grad):

        raise NotImplementedError


# class Model(Module, ABC):
#     def __init__(self, *args, loss=None, optimizer=None):
#         super(Model, self).__init__()
#         self.loss = loss
#         self.optimizer = optimizer
#
#     def step(self, x, y):
#         self.zero_grad()
#         loss = self.backward(x, y)
#         self.apply_grad()
#         return loss
#
#     def train(self, data, n_epochs):
#         x, y = data
#         for _ in range(n_epochs):
#             self.step(x, y)

class Model(Module, ABC):
    def __init__(self, *args, loss=None, optimizer=None):
        super(Model, self).__init__()
        self.loss = loss
        self.optimizer = optimizer

    @abstractmethod
    def train(self, data, n_epochs):
        raise NotImplementedError
