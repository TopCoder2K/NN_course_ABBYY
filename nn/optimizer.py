from src.base import Optimizer


# TODO: add momentum, is_nesterov
class GradientDescend(Optimizer):
    def __init__(self, lr):
        super(GradientDescend, self).__init__()
        raise NotImplementedError  # TODO: replace line with your code

    def step(self, weights, grad):
        raise NotImplementedError  # TODO: replace line with your code
