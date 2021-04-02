import torch
from abc import abstractmethod

from nn.base import Module


class Loss(Module):
    """
    Абстрактный класс для лосса.
    Нужен, так как у лоссов другая семантика вызова ```forward()``` и ```backward()```.
    """

    def __init__(self):
        super(Loss, self).__init__()

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
    """Обычная MSE. Подробности см. в README.md."""

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_pred, y_true):
        self.output = torch.sum(y_pred - y_true)**2 / len(y_true)
        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        self.grad_input = 2. / len(y_true) * (y_pred - y_true)
        return self.grad_input


class CrossEntropy(Loss):
    """
    Функция риска 'перекрёстная энтропия'. В ```forward``` принимает на вход истинные и предсказанные вероятности
    принадлежности к классам. Подробности по формулам см. в README.md.
    """

    EPS = 1e-15  # Для стабильности работы логарифма и деления при нулевых вероятностях.

    def forward(self, y_pred, y_true):
        y_pred_clamp = torch.clip(y_pred, self.EPS, 1 - self.EPS)
        self.output = -torch.sum(torch.log(y_pred_clamp) * y_true) / len(y_pred)

        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        y_pred_clamp = torch.clip(y_pred, self.EPS, 1 - self.EPS)
        self.grad_input = -y_true / y_pred_clamp / len(y_pred)

        return self.grad_input


class KLDivergence(Loss):
    """
    Функция риска 'расстояние Кульбака-Лейблера'. Отличается от кросс-энтропии лишь на энтропию истинного
    распределения вероятностей, поэтому подробных пояснений в README.md нет.
    """

    EPS = 1e-15  # Для стабильности работы логарифма и деления при нулевых вероятностях.

    def forward(self, y_pred, y_true):
        y_pred_clamp = torch.clip(y_pred, self.EPS, 1 - self.EPS)
        y_true_clamp = torch.clip(y_true, self.EPS, 1 - self.EPS)

        self.output = torch.sum(torch.log(y_true_clamp) * y_true) / len(y_pred)
        self.output -= torch.sum(torch.log(y_pred_clamp) * y_true) / len(y_pred)

        return self.output

    def update_module_input_grad(self, y_pred, y_true):
        y_pred_clamp = torch.clip(y_pred, self.EPS, 1 - self.EPS)
        self.grad_input = -y_true / y_pred_clamp / len(y_pred)

        return self.grad_input
