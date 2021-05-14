class SGD:
    """Простой SGD без моментов и регуляризаций."""

    def __init__(self, lr, params):
        """
        Параметры
        ---------
        lr : float
            Шаг оптимизатора.
        params : Dict{str : torch.tensor}
            Словарь параметров модели. Ключи словаря --- названия параметров.
        """

        self.lr = lr
        self.params = params

    def update_params(self, grads):
        for param_name in self.params.keys():
            self.params[param_name] -= self.lr * grads['d' + param_name]

        return self.params
