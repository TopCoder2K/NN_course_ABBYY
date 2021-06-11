from Task2Modules.CNN import Conv, AvgPool
from Task2Modules.OtherLayers import Fc, Flatten, BatchNorm2d
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

        self.conv1 = Conv(nb_filters=1, nb_channels=4, filter_size=input_size,
                          init=init[:2], padding=0, stride=1)
        self.act1 = ReLU()
        self.flatten = Flatten()
        self.fc1 = Fc(4 * 1 * 1, num_classes, init=init[2:4])

        self.layers_with_params = [self.conv1, self.fc1]

    def forward(self, X):
        # print(x.shape)
        X = self.conv1.forward(X)
        # print(x.shape)
        X = self.act1.forward(X)
        # print(x.shape)

        X = self.flatten.forward(X)
        # print(x.shape)
        X = self.fc1.forward(X)
        # print(x.shape)
        # print('============================================================')

        return X

    def backward(self, deltaL):
        dX, dW2, db2 = self.fc1.backward(deltaL)
        # print(dX.shape)
        dX = self.flatten.backward(dX)
        # print(dX.shape)
        dX = self.act1.backward(dX)
        # print(dX.shape)
        dX, dW1, db1 = self.conv1.backward(dX)
        # print(dX.shape)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }

        return grads

    def get_params(self):
        """ Функция для получения словаря параметров модели.

        Returns
        -------
        params : dict
            Словарь из 4 объектов: W и b на свёрточный и линейный слои.
        """

        params = {}
        for i, layer in enumerate(self.layers_with_params):
            params['W' + str(i + 1)] = layer.W
            params['b' + str(i + 1)] = layer.b

        return params

    def set_params(self, params):
        """ Сеттер лдя параметров модели.
        
        Parameters
        ----------
        params : dict
            Словарь, содержащий 4 элемента: W и b на свёрточный и линейный слои.
        """
        
        for i, layer in enumerate(self.layers_with_params):
            layer.W = params['W' + str(i + 1)]
            layer.b = params['b' + str(i + 1)]


class CustomModel2:
    def __init__(self, input_size=5, num_classes=2):
        self.conv1 = Conv(nb_filters=1, nb_channels=4, filter_size=3, padding=0)
        H = W = (input_size - 3) + 1  # по формуле преобразования размеров

        self.bn1 = BatchNorm2d(4, alpha=0.9)
        self.act1 = ReLU()
        self.pool1 = AvgPool(filter_size=2, stride=2)
        new_H = new_W = int(1 + (H - 2) / 2)  # тоже по формуле

        self.flatten = Flatten()
        new_H, new_W = 1, 4 * new_H * new_W

        self.fc1 = Fc(new_W, num_classes)

        self.layers_with_params = [self.conv1, self.bn1, self.fc1]

    def forward(self, X):
        # (batch_size, 4, H, W)
        X = self.act1.forward(self.bn1.forward(self.conv1.forward(X)))
        # (batch_size, 4, new_H, new_W)
        X = self.pool1.forward(X)
        # (batch_size, num_classes)
        X = self.fc1.forward(self.flatten.forward(X))

        return X

    def backward(self, deltaL):
        dX, dW3, db3 = self.fc1.backward(deltaL)
        dX = self.flatten.backward(dX)
        dX = self.act1.backward(self.pool1.backward(dX))
        dX, dW2, db2 = self.bn1.backward(dX)
        dX, dW1, db1 = self.conv1.backward(dX)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }

        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers_with_params):
            params['W' + str(i + 1)] = layer.W
            params['b' + str(i + 1)] = layer.b

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers_with_params):
            layer.W = params['W' + str(i + 1)]
            layer.b = params['b' + str(i + 1)]
