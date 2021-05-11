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

        # Не знаю, к чему комментарий выше, ведь мы легко это учтём в самом
        # классе CrossEntropy
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


# class BatchNormalization:
#     """ Батч нормализация без скейлинга. """
#
#     EPS = 1e-3
#
#     def __init__(self, alpha=0.):
#         self.alpha = alpha
#         self.moving_mean = None
#         self.moving_variance = None  # Будем хранить именно \sigma^2
#
#     def forward(self, layer_input):
#         widehat_mu = torch.mean(layer_input, dim=0)
#         widehat_sigma = torch.mean((layer_input - widehat_mu) ** 2, dim=0)
#
#         if self.moving_mean is None:
#             self.moving_mean = 0.
#         if self.moving_mean is None:
#             self.moving_variance = 0.
#
#         # TODO: условие некорректное????????????????????????????????
#         if self.training == True:
#             self.moving_mean = self.alpha * self.moving_mean + \
#                                widehat_mu * (1 - self.alpha)
#             self.moving_variance = self.alpha * self.moving_variance + \
#                                    widehat_sigma * (1 - self.alpha)
#             self.output = (input - widehat_mu) / \
#                           (widehat_sigma + BatchNormalization.EPS) ** 0.5
#         else:
#             self.output = (input - self.moving_mean) / \
#                           (self.moving_variance + BatchNormalization.EPS) ** 0.5
#
#         return self.output
#
#     def backward(self, input, grad_output):
#         N = np.asarray(input).shape[0]
#         widehat_mu = np.mean(input, axis=0)
#         widehat_sigma = np.mean((input - widehat_mu) ** 2, axis=0)
#
#         self.grad_input = grad_output / (widehat_sigma + BatchNormalization.EPS) ** 0.5 + \
#                           -1. / (widehat_sigma + BatchNormalization.EPS) ** 0.5 * np.mean(grad_output, axis=0) + \
#                           -0.5 * (widehat_sigma + BatchNormalization.EPS) ** (-3. / 2) * \
#                           np.sum(grad_output * (input - widehat_mu), axis=0) * 2 * (input - widehat_mu) / N
#
#         return self.grad_input
#
#
# class Scaling:
#     """ Скейлинг с обучаемыми параметрами. """
#
#     def __init__(self, n_out):
#         stdv = 1. / np.sqrt(n_out)
#         self.gamma = np.random.uniform(-stdv, stdv, size=(1, n_out))
#         self.beta = np.random.uniform(-stdv, stdv, size=(1, n_out))
#
#         self.gradGamma = np.zeros_like(self.gamma)
#         self.gradBeta = np.zeros_like(self.beta)
#
#     def update_output(self, input):
#         """
#         Вход:
#             `input (np.array)` -- вход слоя
#         """
#         self.output = input * self.gamma + self.beta
#         return self.output
#
#     def update_grad_input(self, input, grad_output):
#         """
#         Вход:
#             `input (np.array)` -- вход слоя
#             `grad_output (np.array)` -- градиент по выходу этого слоя, пришедший от следующего слоя
#         """
#         self.grad_input = np.multiply(grad_output, self.gamma)
#         return self.grad_input
#
#     def update_grad_params(self, input, grad_output):
#         """
#         Вход:
#             `input (np.array)` -- вход слоя
#             `grad_output (np.array)` -- градиент по выходу этого слоя, пришедший от следующего слоя
#         """
#         self.gradBeta = np.sum(grad_output, axis=0)
#         self.gradGamma = np.sum(np.multiply(grad_output, input), axis=0)
#
#     def zero_grad_params(self):
#         self.gradGamma.fill(0)
#         self.gradBeta.fill(0)
#
#     def get_parameters(self):
#         return [self.gamma, self.beta]
#
#     def get_grad_params(self):
#         return [self.gradGamma, self.gradBeta]
#
#
# class BatchNorm2d:
#     """ Батч нормализацию можно разбить на два этапа: нормализация и скейлинг. """
#
#     def __init__(self, num_channels, gamma=1, beta=0, eps=1e-20):
#         self.num_channels = num_channels
#         # Полная фигня, но применяем стандартные название полей для обновления весов,
#         # чтобы не переписывать код модели и оптимизатора
#         self.W = torch.ones(num_channels)  # gamma
#         self.b = torch.zeros(num_channels)  # beta
#         self.eps = eps
#
#         self.dW = torch.zeros(num_channels)
#         self.db = torch.zeros(num_channels)
#
#         self.cache = None
#
#     def forward(self, x, debug=True):
#         raise NotImplementedError
#
#     def backward(self, dout):
#         raise NotImplementedError
