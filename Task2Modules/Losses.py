import torch
from Task2Modules.OtherLayers import LogSoftmax


class CrossEntropyLoss:
    def __init__(self):
        self.log_sm = LogSoftmax()

    # TODO: Александр, а где же общие интерфейсы и соответствие им?((
    def get(self, y_pred, y):
        """
        Считает лосс и градиент лосса.

        Параметры
        ---------
        y_pred : torch.tensor, shape = (batch_size, n_classes)
            Предсказанные вероятности принадлежности классам.
        y : torch.tensor, shape = (batch_size, n_classes)
            One-hot encoding истинных меток классов.
        """

        # forward
        log_probs = self.log_sm.forward(y_pred)
        # TODO: нужно на число элементов в y_true???????????
        cross_entropy = -torch.sum(torch.mul(log_probs, y)) / y.shape[0]

        # backward
        dL = self.log_sm.backward(-y / y.shape[0])

        return cross_entropy, dL
