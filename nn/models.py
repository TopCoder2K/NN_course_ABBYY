from nn.base import Model


class FeedForwardModel(Model):
    """
    Этот класс является последовательностью слоёв.
    Последовательно обрабатывает вход `model_input` от слоя к слою.

    Аттрибуты
    ---------
    `layers` : list
        Список всех слоёв модели.
    `loss` : callable
        Функция риска, принимает y_true, y_predicted, возвращает число.
    `optimizer` : class
        Класс, унаследованный от `Optimizer`. Выполняет корректировку весов на каждой эпохе.
    `output` : torch.tensor
        Выход всей модели. Нужен для того, чтобы передавать в loss?????????????????????????? TODO
    `grad_input` : torch.tensor
        Градиент функции риска по входу модели (Нужен для удобства).
    """

    def __init__(self, layers=None, loss=None, optimizer=None):
        super(FeedForwardModel, self).__init__(loss=loss, optimizer=optimizer)
        self.layers = layers

    def forward(self, model_input):
        """
        Делает прямой проход по всем слоям.

        Параметры
        ----------
        `model_input` : torch.tensor
            Входные данные.
        Возвращает
        ----------
        torch.tensor
            Выход модели.
        """

        output = model_input.copy()

        for layer in self.layers:
            output = layer.forward(output)
            layer.output = output

        self.output = output    # TODO: а это вообще нужно?
        return output

    def backward(self, model_input, grad_output):
        """
        Делает обратный проход по всем слоям, передавая каждому слою градиент по выходу этого слоя.

        Параметры
        ----------
        `model_input` : torch.tensor
            Входные данные модели.
        `grad_output` : torch.tensor
            Градиент функции потерь по выходу сети.
        Возвращает
        ----------
        torch.tensor
            Градиент лосса по весам модели.
        """

        g = grad_output.copy()

        # Проходим с конца до 0-го НЕвключительно
        for i in range(1, len(self.layers)):
            g = self.layers[-i].backward(self.layers[-i - 1].output, g)

        self.grad_input = self.layers[0].backward(input, g)
        return self.grad_input

    def zero_grad(self):
        """Зануляет градиенты у всех слоёв."""

        for layer in self.layers:
            layer.zero_grad_params()

    @property
    def parameters(self):
        """Собирает параметры каждого слоя в один список, получая список списков."""

        return [layer.parameters for layer in self.layers]

    @property
    def grad_params(self):
        """Собирает градиенты параметров каждого слоя в один список, получая список списков."""

        return [layer.grad_params for layer in self.layers]

    # def apply_grad(self):
    #     raise NotImplementedError  # TODO: replace line with your implementation

    def train(self, data_train, n_epochs):
        """
        Функция для обучения модели. По-хорошему должна принимать ещё батч генератор, но пока без него.

        Параметры
        ---------
        `data_train` : torch.tensor
            Набор данных для обучения.
        `n_epochs` : integer
            Число эпох обучения.
        Возращает
        ---------
        `loss_history` : list
            Все значения лосса в процессе обучения.
        """

        x_train, y_train = data_train
        loss_history = []

        for _ in range(n_epochs):
            # Обнуляем градиенты с предыдущей итерации
            self.zero_grad()
            # Forward pass
            y_pred = self.forward(x_train)
            loss = self.loss.forward(y_pred, y_train)

            # Backward pass
            last_grad_output = self.loss.backward(y_pred, y_train)
            self.backward(x_train, last_grad_output)

            # Обновление весов
            self.optimizer.step(self.parameters, self.grad_params)

            loss_history.append(loss)

        return loss_history
