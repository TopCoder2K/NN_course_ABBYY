from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import numpy as np


class Module(ABC):
    """
    Абстрактный класс модуля. Именно из модулей будет состоять нейронная сеть.

    Атрибуты
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
        pass

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

        self.update_module_input_grad(module_input, grad_output)
        self.update_params_grad(module_input, grad_output)
        return self.grad_input

    # TODO: нужно ли возвращать градиент?
    def update_module_input_grad(self, module_input, grad_output):
        """
        Вычисляет градиент функции слоя по входу и возвращает его в виде `self.grad_input`.
        Размер (`shape`) поля `self.grad_input` всегда совпадает с размером `input`.
        Тоже не абстрактный, так как для самой модели он не нужен.

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
        pass

    def update_params_grad(self, module_input, grad_output):
        """
        Вычисляет градиент функции слоя по параметрам (весам) этого слоя.
        Ничего не возвращает, только сохраняет значения градиентов в соответствующие поля.
        Так как не у всех слоёв есть параметры, этот метод необязательный к переопределению.

        Параметры
        ---------
        `module_input` : torch.tensor
            Вход слоя
        `grad_output` : torch.tensor
            Градиент по выходу этого слоя, пришедший от следующего слоя
        """
        pass

    def zero_grad(self):
        """Зануляет градиенты у параметров слоя (если они есть). Нужно для оптимизатора."""
        pass

    @property
    def parameters(self):
        """
        Возвращает список параметров этого слоя, если они есть. Иначе возвращает пустой список.
        Нужно для оптимизатора.
        """

        return []

    @property
    def grad_params(self):
        """
        Возвращает список градиентов функции этого слоя по параметрам этого слоя, если они есть. 
        Иначе возвращает пустой список. Нужно для оптимизатора.
        """

        return []

    # @abstractmethod
    # def apply_grad(self):
    #     """?????????????????????????"""
    #     raise NotImplementedError


class Optimizer(ABC):
    """
    Абстрактный класс оптимизатора.

    Атрибуты
    --------
    `config` : dict
        Словарь c гиперпараметрами оптимизатора. Дефолтно имеет три гиперпараметра: learning rate,
        параметр L1-регуляризации, параметр L2-регуляризации.
    `state` : dict
        Словарь cо состоянием оптимизатора. Нужен, чтобы сохранять старые значения градиентов.
    """

    def __init__(self, lr=0.01, l1=None, l2=None):
        """
        Параметры
        ---------
        `lr` : float
            Шаг обучения.
        `l1` : float
            Параметр L1-регуляризации.
        `l2` : float
            Параметр L2-регуляризации.
        """

        self.config = defaultdict(float)
        self.config['lr'] = lr
        self.config['l1'] = l1
        self.config['l2'] = l2
        self.state = {}

    def _add_regularization_grad(self, params, params_grad):
        """
        Если задана регуляризация, изменяет градиенты параметров соответствующим образом.
        Так как вся логика работы с регуляризацией будет в оптимизаторе, значения лосса будут как бы ненастоящими,
        но нам и не важно, так как обычно играют роль относительные изменения лосса.

        Параметры
        ---------
        `params` : torch.tensor
            Параметры модели.
        `params_grad` : torch.tensor
            Градиент функции потерь по параметрам модели.
        """

        if self.config['l1'] is not None:
            params_grad += self.config['l1'] * torch.sgn(params)

        if self.config['l2'] is not None:
            params_grad += 2 * self.config['l2'] * params

    @abstractmethod
    def step(self, params, params_grad):
        """
        Делает один шаг в соответствии с алгоритмом оптимизации.

        Параметры
        ---------
        `params` : torch.tensor
            Параметры (веса) модели.
        `params_grad` : torch.tensor
            Градиент функции риска по параметрам (весам) модели.
        """
        pass


class Model(Module, ABC):
    def __init__(self, *args, loss=None, optimizer=None):
        super(Model, self).__init__()
        self.loss = loss
        self.optimizer = optimizer

    @staticmethod
    def _train_batch_gen(x_train, y_train, batch_size):
        """
        Генератор батчей.
        На каждом шаге возвращает `batch_size` объектов из `x_train` и их меток из `labels`.

        Параметры
        ---------
        `x_train` : torch.tensor
            Признаковые описания объектов.
        `y_train` : torch.tensor
            Значения целевой переменной.
        `batch_size` : integer
            Размер батча.
        Возвращает
        ----------
        Пару тензоров --- текущий батч объектов из x_train и y_train.
        """

        n_samples = x_train.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Перемешиваем в случайном порядке в начале эпохи

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            yield x_train[batch_idx], y_train[batch_idx]

    @abstractmethod
    def train(self, data_train, n_epochs, batch_size=64):
        """
        Функция для обучения модели. По-хорошему должна принимать ещё батч генератор, но пока без него.

        Параметры
        ---------
        `data_train` : array like, shape=(1, 2)
            Набор данных для обучения. Содержит в себе 2 тензора: признаковые описания и целевую переменную.
        `n_epochs` : integer
            Число эпох обучения.
        `batch_size` : integer
            Размер батча во время обучения.
        Возращает
        ---------
        `loss_history` : list
            Все значения лосса в процессе обучения.
        """
        pass
