from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def backward(self, x, grad_output):  # compute grad
        raise NotImplementedError

    def zero_grad(self):
        pass

    def apply_grad(self):
        pass


class Optimizer(ABC):
    def __init__(self, *args):
        pass

    @abstractmethod
    def step(self, weights, grad):
        raise NotImplementedError


class Model(Module, ABC):
    def __init__(self, *args, loss=None, optimizer=None):
        super(Model, self).__init__()
        self.loss = loss
        self.optimizer = optimizer

    def step(self, x, y):
        self.zero_grad()
        loss = self.backward(x, y)
        self.apply_grad()
        return loss

    def train(self, data, n_epochs):
        x, y = data
        for _ in range(n_epochs):
            self.step(x, y)
