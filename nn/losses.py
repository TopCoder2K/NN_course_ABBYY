import torch
import numpy as np
from abc import abstractmethod

from nn.base import Module
from nn.layers import LogSoftmax


class Loss(Module):
    """
    Абстрактный класс для лосса.
    Нужен, так как у лоссов другая семантика вызова `forward()` и `backward()`.

    Атрибуты
    --------
    `output` : torch.tensor
        Выход лосса.
    `grad_input` : torch.tensor
        Градиент по входу функции риска.
    """

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        Считает нужную функцию риска.

        Параметры
        ---------
        y_true : torch.tensor
            Истинное значение целевой переменной.
        y_pred : torch.tensor
            Предсказанное значение целевой переменной.
        Возвращает
        ----------
        float
            Значение функции риска.
        """
        pass

    def backward(self, y_pred, y_true):
        """
        По аналитической формуле считает значение градиента функции потерь при заданных y_true и y_pred.

        Параметры
        ---------
        y_true : torch.tensor
            Истинное значение целевой переменной.
        y_pred : torch.tensor
            Предсказанное значение целевой переменной.
        Возвращает
        ----------
        torch.tensor
            Градиент по входу модуля, а именно градиент по y_pred.
        """

        self.update_module_input_grad(y_pred, y_true)
        return self.grad_input

    @abstractmethod
    def update_module_input_grad(self, y_pred, y_true):
        pass


class MSE(Loss):
    """Обычная MSE. Подробности по формулам см. в README.md.

    Атрибуты
    --------
    `output` : torch.tensor
        Выход лосса.
    `grad_input` : torch.tensor
        Градиент по входу функции риска.
    """

    def forward(self, y_pred, y_true):
        self.output = torch.sum((y_pred - y_true) ** 2) / y_true.nelement()

        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        self.grad_input = 2. / y_true.nelement() * (y_pred - y_true)

        return self.grad_input


class CrossEntropy(Loss):
    """
    Функция риска 'перекрёстная энтропия'. В `forward()` принимает на вход необработанные скоры для каждого из классов
    и истинные метки классов. Подробности по формулам см. в README.md.

    Атрибуты
    --------
    `output` : torch.tensor
        Выход лосса.
    `grad_input` : torch.tensor
        Градиент по входу функции риска.
    """

    EPS = 1e-15  # Для стабильности работы логарифма и деления при нулевых вероятностях.

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.log_softmax_layer = LogSoftmax()

    def forward(self, y_pred, y_true):
        batch_size, n_in = y_pred.shape
        target = torch.zeros((batch_size, n_in))
        target[np.arange(batch_size), y_true] = 1  # one-hot encoding

        log_probs = self.log_softmax_layer.forward(y_pred)
        # Заметим, что ниже делить нужно на число элементов в y_true!
        self.output = -torch.sum(torch.mul(log_probs, target)) / y_true.nelement()

        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        batch_size, n_in = y_pred.shape
        target = torch.zeros((batch_size, n_in))
        target[np.arange(batch_size), y_true] = 1  # one-hot encoding

        self.grad_input = self.log_softmax_layer.backward(y_pred, -target / y_true.nelement())

        return self.grad_input


class KLDivergence(Loss):
    """
    Функция риска 'расстояние Кульбака-Лейблера'. Отличается от кросс-энтропии лишь на энтропию истинного
    распределения вероятностей, поэтому подробных пояснений по формулам в README.md нет.
    Также есть отличия в `forward()`: на вход ожидаются уже логарифмы предсказанных вероятностей, а также истинные
    вероятности.
    Усреднение происходит по батчам.

    Атрибуты
    --------
    `output` : torch.tensor
        Выход лосса.
    `grad_input` : torch.tensor
        Градиент по входу функции потерь.
    """

    EPS = 1e-15  # Для стабильности работы логарифма и деления при нулевых вероятностях.

    def forward(self, y_pred, y_true):
        y_true_clamp = torch.clip(y_true, self.EPS, 1 - self.EPS)
        self.output = torch.sum(torch.mul(torch.log(y_true_clamp) - y_pred, y_true)) / len(y_pred)

        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        self.grad_input = -y_true / len(y_pred)

        return self.grad_input
