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
        new_params = self.params
        #         print(grads)

        new_params['W1'] -= self.lr * grads['dW1']
        new_params['b1'] -= self.lr * grads['db1']
        new_params['W2'] -= self.lr * grads['dW2']
        new_params['b2'] -= self.lr * grads['db2']

        return new_params
