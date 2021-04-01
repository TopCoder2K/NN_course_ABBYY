import torch

from nn.base import Module


class Sigmoid(Module):
    """Сигмоидная функция активации. Подробности см. в README.md."""

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, module_input):
        # Заметим, что при больших значениях аргумента переполняться будет e^x, а при сильно отрицательных значениях
        # --- e^{-x}, поэтому имеем смысл сделать разветвление.
        positive = module_input >= 0
        negative = ~positive

        self.output = torch.empty(module_input.shape)
        self.output[positive] = 1. / (1. + torch.exp(-module_input[positive]))
        self.output[negative] = torch.exp(module_input[negative]) / (1. + torch.exp(module_input[positive]))
        return self.output

    def update_module_input_grad(self, module_input, grad_output):
        return grad_output.mul(self.output.mul(1 - self.output))


class ReLU(Module):
    """Функция активации ReLU. Подробности см. в README.md."""

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, module_input):
        self.output = torch.maximum(module_input, torch.zeros(size=module_input.shape))
        return self.output

    def update_module_input_grad(self, layer_input, grad_output):
        self.grad_input = torch.mul(grad_output, layer_input > 0)
        return self.grad_input
