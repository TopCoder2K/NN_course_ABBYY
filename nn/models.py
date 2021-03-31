from nn.base import Model


class FeedForwardModel(Model):
    """
    Этот класс является последовательностью слоёв.
    Последовательно обрабатывает вход 'x' от слоя к слою.

    Аттрибуты
    ---------
    layers : list
        Список всех слоёв модели
    loss : callable
        Функция риска, принимает y_true, y_predicted, возвращает число
    optimizer : class
        Класс, унаследованный от `Optimizer`. Выполняет корректировку весов на каждой эпохе
    output : torch.tensor
        Выход всей модели. Нужен для того, чтобы передавать в loss?????????????????????????? TODO
    grad_input : torch.tensor
        Градиент функции риска по входу модели (Нужен для удобства).
    """

    def __init__(self, layers=None, loss=None, optimizer=None):
        super(FeedForwardModel, self).__init__(loss=loss, optimizer=optimizer)
        self.layers = layers

    def forward(self, x):
        """Делает прямой проход по всем слоям.

        Параметры
        ----------
        x : torch.tensor, shape = (n, 1)
            Входные данные
        Возвращает
        ----------
        torch.tensor
            Выход модели
        """
        output = x.copy()
        for layer in self.layers:
            output = layer.forward(output)
            layer.output = output

        self.output = output    # TODO: а это вообще нужно?
        return output

    def backward(self, x, grad_output):

        g = grad_output.copy()

        # Проходим с конца до 0-го НЕвключительно
        for i in range(1, len(self.layers)):
            g = self.layers[-i].backward(self.layers[-i - 1].output, g)

        self.grad_input = self.layers[0].backward(input, g)
        return self.grad_input

    def zero_grad(self):
        """Зануляет градиенты у всех слоёв"""
        for layer in self.layers:
            layer.zero_grad_params()

    def apply_grad(self):
        raise NotImplementedError  # TODO: replace line with your implementation

    def train(self, data, n_epochs):
        x, y = data
        for _ in range(n_epochs):
            # Обнуляем градиенты с предыдущей итерации
            self.zero_grad()
            # Forward pass
            y_pred = self.forward(x_batch)
            loss = criterion.forward(y_pred, y_batch)
            # Backward pass
            last_grad_input = criterion.backward(y_pred, y_batch)
            model.backward(x_batch, last_grad_input)
            # Обновление весов
            SGD(model.get_parameters(),
                model.get_grad_params(),
                opt_params,
                opt_state)
            loss_history.append(loss)
        # raise NotImplementedError  # TODO: replace line with your implementation
