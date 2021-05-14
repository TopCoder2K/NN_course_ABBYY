import torch


class Fc:
    """ Полносвязный слой. """

    def __init__(self, in_features, out_features, init=None):
        """
        Конструктор слоя. Заметим, что bias всегда есть.

        Параметры
        ---------
        in_features : int
            Число фич у входа слоя.
        out_features : int
            Число фич у выхода слоя.
        init : tuple of torch.tensors
            Содержит два тензора: первый --- для инициализации тензора весов,
            второй --- для инициализации тензора bias.
        """

        self.W = torch.zeros((in_features, out_features))
        self.b = torch.zeros((out_features,))

        if init is not None:
            self.W = init[0].t()
            self.b = init[1]
        else:
            self.W = torch.nn.init.uniform_(self.W)
            self.b = torch.nn.init.uniform_(self.b)

        self.dW = torch.zeros(self.W.shape)
        self.db = torch.zeros(self.b.shape)

        self.cache = {}

    def forward(self, X):
        self.cache['input'] = X

        self.cache['output'] = torch.matmul(X, self.W)
        self.cache['output'] += self.b

        return self.cache['output']

    def backward(self, dZ):
        # не забываем, что для эмуляции CrossEntropyLoss(reduction='mean')
        # нужно делить self.dW и self.db на размер пакета

        # Не знаю, к чему преподавательский комментарий выше,
        # ведь мы легко это учтём в самом классе CrossEntropy
        self.dW = torch.matmul(self.cache['input'].t(), dZ)
        self.db = torch.sum(dZ, dim=0)

        return torch.matmul(dZ, self.W.t()), self.dW, self.db


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        # Нормализуем для численной устойчивости, а потом возводим в exp
        self.output = torch.exp(X - X.max(dim=1, keepdim=True).values)
        self.output /= self.output.sum(dim=1, keepdim=True)

        return self.output

    def backward(self, dZ):
        grad_input = torch.zeros(size=self.output.shape)

        for i in range(self.output.shape[0]):
            softmax_i = self.output[i, :].unsqueeze(1)
            partial_softmax = -torch.matmul(softmax_i, softmax_i.t()) + \
                              torch.diag(softmax_i.squeeze())
            for j in range(self.output.shape[1]):
                grad_input[i, j] = torch.dot(dZ[i, :], partial_softmax[:, j])

        return grad_input


class LogSoftmax:
    """Осуществляет log(softmax)-преобразование."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.detach().clone()

        # Нормализуем для численной устойчивости
        self.output = X - X.max(axis=1, keepdim=True).values
        self.output = self.output - torch.log(torch.sum(torch.exp(self.output),
                                                        dim=1, keepdim=True))

        return self.output

    def backward(self, dZ):
        # Нормализуем для численной устойчивости, а потом уже возводим в exp
        exp_module_input = torch.exp(
            self.input - self.input.max(axis=1, keepdim=True).values
        )
        softmax = exp_module_input / torch.sum(exp_module_input, dim=1,
                                               keepdim=True)

        grad_input = dZ - torch.mul(softmax, torch.sum(dZ, dim=1, keepdim=True))

        return grad_input


class Flatten:
    """ Убирает все размерности, кроме нулевой (размерность батча). """

    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(len(X), -1)

    def backward(self, dZ):
        return dZ.reshape(self.input_shape)


class BatchNormalization2D:
    """ 2D батч нормализация без скейлинга.
    Подробности по формулам в ноутбуке.
    """

    def __init__(self, nb_channels, alpha=0., eps=1e-3):
        self.alpha = alpha
        self.moving_mean = torch.zeros((1, nb_channels, 1, 1))
        # Будем хранить именно \sigma^2
        self.moving_var = torch.zeros((1, nb_channels, 1, 1))

        self.training = True

        self.eps = eps
        self.cache = {}

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, X):
        mu = torch.mean(X, dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        sigma2 = torch.mean((X - mu) ** 2, dim=(0, 2, 3), keepdim=True)

        self.cache['input'] = X.detach().clone()
        self.cache['mu'] = mu.detach().clone()
        self.cache['sigma2'] = sigma2.detach().clone()

        if self.training:
            self.moving_mean = self.alpha * self.moving_mean + \
                               mu * (1 - self.alpha)
            self.moving_var = self.alpha * self.moving_var + \
                              sigma2 * (1 - self.alpha)
            output = (X - mu) / (sigma2 + self.eps) ** 0.5
        else:
            output = (X - self.moving_mean) / \
                     (self.moving_var + self.eps) ** 0.5

        return output

    def backward(self, dZ):
        X, mu = self.cache['input'], self.cache['mu']
        sigma2 = self.cache['sigma2']
        N, C, H, W = X.shape
        dX = torch.zeros((N, C, H, W))

        for c in range(C):
            mu_c = mu[0, c, 0, 0]
            sigma2_c = sigma2[0, c, 0, 0]

            grad_sigma2_c = torch.sum(torch.mul(
                dZ[:, c, :, :],
                -0.5 * (X[:, c, :, :] - mu_c) * (sigma2_c + self.eps) ** (
                            -3 / 2)
            ), dim=(0, 1, 2))
            grad_mu_c = torch.sum(torch.mul(
                dZ[:, c, :, :], -1 / (sigma2_c + self.eps) ** 0.5
            ), dim=(0, 1, 2))
            dX[:, c, :, :] = \
                dZ[:, c, :, :] / (sigma2_c + self.eps) ** 0.5 + \
                grad_mu_c / (N * H * W) + \
                grad_sigma2_c * 2 * (X[:, c, :, :] - mu_c) / (N * H * W)

        return dX


class Scaling:
    """ Скейлинг в 2D с обучаемыми параметрами gamma и beta из R^{1, C, 1, 1}.
    Подробности по формулам в ноутбуке.
    """

    def __init__(self, nb_channels):
        self.gamma = torch.ones((1, nb_channels, 1, 1))
        self.beta = torch.zeros((1, nb_channels, 1, 1))
        self.cache = {}

    def forward(self, X):
        self.cache['input'] = X.detach().clone()
        return torch.mul(X, self.gamma) + self.beta  # broadcasting должен
        # сработать как надо

    def backward(self, dZ):
        """ Обратный проход для слоя.
        Так как мы работаем в немасштабируемой, но простой концепции, когда
        `backward()` каждого слоя возвращает все градиенты, слой Scaling-а
        тоже должен вернуть как градиенты по входу, так и по параметрам.

        Parameters
        ----------
        dZ : torch.tensor, shape = (N, C, H, W)
            Градиент лосса по выходу слоя.

        Returns
        -------
        dX : torch.tensor, shape = (N, C, H, W)
            Градиент лосса по входу слоя.
        dGamma : torch.tensor, shape = (1, C, 1, 1)
            Градиент лосса по параметру масштаба.
        dBeta : torch.tensor, shape = (1, C, 1, 1)
            Градиент лосса по параметру сдвига.
        """

        dX = torch.mul(dZ, self.gamma)  # (N, C, H, W)

        dGamma = torch.sum(torch.mul(dZ, self.cache['input']),
                           dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
        dBeta = torch.sum(dZ, dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)

        return dX, dGamma, dBeta


class BatchNorm2d:
    """ Батч нормализацию можно разбить на два этапа: нормализация и скейлинг.
    Здесь каждый из этапов применятся по очереди.
    """

    def __init__(self, nb_channels, alpha, eps=1e-5):
        self.eps = eps
        self.batch_norm = BatchNormalization2D(nb_channels, alpha, eps)
        self.scale = Scaling(nb_channels)

        # Так как структура нормальная не разработана, костылим
        self.W = self.scale.gamma
        self.b = self.scale.beta

    def train(self):
        self.batch_norm.train()

    def eval(self):
        self.batch_norm.eval()

    def forward(self, X):
        return self.scale.forward(self.batch_norm.forward(X))

    def backward(self, dZ):
        return self.scale.backward(self.batch_norm.backward(dZ))
