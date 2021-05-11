from Task2Modules.CNN import Conv
from Task2Modules.OtherLayers import Fc, Flatten
from Task2Modules.Activations import ReLU


class CustomModel:
    """ Простейшая модель с одним свёрточным слоём. """

    def __init__(self, init, input_size=5, num_classes=2):
        """
        Конструктор модели.

        Параметры
        ---------
        init : array of torch.tensors
            Содержит 4 тензора: 2 тензора для инициализации свёрточного слоя
            и 2 тензора --- для полносвязного слоя.
        input_size : int
            Размер выходного изображения. По умолчанию используется для размера
            фильтра в свёрточном слое.
        num_classes : int
            Число классов в классификации.
        """

        self.conv1 = Conv(nb_filters=1, filter_size=input_size, init=init[:2],
                          nb_channels=4, padding=0, stride=1)
        self.act1 = ReLU()
        self.flatten = Flatten()
        self.fc1 = Fc(4 * 1 * 1, num_classes, init=init[2:4])

        self.layers = [self.conv1, self.fc1]

    def forward(self, X):
        #         print(x.shape)
        X = self.conv1.forward(X)
        #         print(x.shape)
        X = self.act1.forward(X)
        #         print(x.shape)

        X = self.flatten.forward(X)
        #         print(x.shape)
        X = self.fc1.forward(X)
        #         print(x.shape)
        #         print('============================================================')

        return X

    def backward(self, deltaL):
        dX, dW2, db2 = self.fc1.backward(deltaL)
        #         print(dX.shape)
        dX = self.flatten.backward(dX)
        #         print(dX.shape)
        dX = self.act1.backward(dX)
        #         print(dX.shape)
        dX, dW1, db1 = self.conv1.backward(dX)
        #         print(dX.shape)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }

        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W' + str(i + 1)] = layer.W
            params['b' + str(i + 1)] = layer.b

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W = params['W' + str(i + 1)]
            layer.b = params['b' + str(i + 1)]


# class CustomModel2:
#     def __init__(self, input_size=5, num_classes=2):
#         self.conv1 = Conv(1, 4, filter_size=3, padding=0)
#         self.bn1 = BatchNorm2d(4)
#         self.act1 = ReLU()
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(4 * int((input_size - 2) / 2) * int((input_size - 2) / 2), num_classes)
#
#         self.layers = [self.conv1, self.bn1, self.fc1]
#
#     def forward(self, x):
#         raise NotImplementedError
#
#     def backward(self, deltaL):
#         raise NotImplementedError
#
#     def get_params(self):
#         params = {}
#         for i, layer in enumerate(self.layers):
#             params['W' + str(i + 1)] = layer.W
#             params['b' + str(i + 1)] = layer.b
#
#         return params
#
#     def set_params(self, params):
#         for i, layer in enumerate(self.layers):
#             layer.W = params['W' + str(i + 1)]
#             layer.b = params['b' + str(i + 1)]
