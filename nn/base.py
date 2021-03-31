from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        self.output = None
        self.grad_input = None

    @abstractmethod
    def forward(self, *args):
        """Вычисляет операцию слоя."""
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def backward(self, x, grad_output):  # compute grad
        """Осуществляет шаг backpropagation'а для этого слоя,
        дифференцируя функцию слоя по входу и по параметрам.

        Параметры
        ---------
        x : torch.tensor
            Вход слоя
        grad_output : torch.tensor
            Градиент по выходу этого слоя, пришедший от следующего слоя
        Возвращает
        ----------
        self.grad_input : torch.tensor
            Градиент функции слоя по входу. Сделано для удобства, так как при обратном распространении ошибки
            каждому слою нужен градиент по выходу этого слоя. Но выход этого слоя есть вход для следующего слоя,
            поэтому логично возвращать градиент по входу.
        """
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self):

        raise NotImplementedError

    @abstractmethod
    def apply_grad(self):
        raise NotImplementedError


class Optimizer(ABC):
    def __init__(self, *args):
        pass

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
