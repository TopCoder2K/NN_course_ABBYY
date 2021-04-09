import torch
import numpy as np
import unittest

from nn.layers import FullyConnectedLayer, Softmax, LogSoftmax
from nn.models import FeedForwardModel
from nn.activations import ReLU, Sigmoid
from nn.losses import MSE, CrossEntropy, KLDivergence
from nn.optimizer import GradientDescend

RANDOM_SEED = 1


class TestLayers(unittest.TestCase):
    """Класс для тестирования всех модулей."""

    @staticmethod
    def _generate_test_data(shape, minval=-10, maxval=10):
        """Генерирует тестовые данные для `forward()` и `backward()`."""

        rand_array = torch.distributions.uniform.Uniform(minval, maxval).sample(shape)
        # Иногда имеет смысл нормализовать
        # rand_array /= rand_array.sum(axis=-1, keepdims=True)
        # rand_array = rand_array.clip(1e-5, 1.)
        # rand_array = 1. / rand_array
        return rand_array

    @staticmethod
    def _custom_forward_backward(layer_input, next_layer_grad, custom_layer, return_params_grad=False):
        """
        Вычисляет результат `forward()` и `backward()` в слое `layer` для небиблиотечной реализации.

        Параметры
        ---------
        `layer_input` : torch.tensor
            Тестовый вход
        `next_layer_grad` : torch.tensor
            Тестовый градиент, пришедший от следующего слоя
        `layer` : class inherited from `Module`
            Слой из нашего мини-фреймворка
        `return_params_grad` : bool
            Если True, то вернуть еще градиенты параметров слоя
        Возвращает
        ----------
        `custom_layer_output` : torch.tensor
            Выход слоя `layer` после `forward()`
        `custom_layer_input_grad` : torch.tensor
            Градиент слоя `layer` после `backward()`
        `custom_params_grad` : torch.tensor [optional]
            Градиенты параметров слоя `layer`
        """

        custom_layer_output = custom_layer.forward(layer_input)
        custom_layer_input_grad = custom_layer.backward(layer_input, next_layer_grad)
        if return_params_grad:
            custom_layer.update_params_grad(layer_input, next_layer_grad)
            custom_params_grad = custom_layer.grad_params
            return custom_layer_output, custom_layer_input_grad, custom_params_grad
        else:
            return custom_layer_output, custom_layer_input_grad

    @staticmethod
    def _torch_forward_backward(layer_input, next_layer_grad, torch_layer, return_params_grad=False):
        """
        Вычисляет результат `forward()` и `backward()` в PyTorch-слое `layer`.

        Параметры
        ---------
        `layer_input` : torch.tensor
            Тестовый вход
        `next_layer_grad` : torch.tensor
            Тестовый градиент, пришедший от следующего слоя
        `torch_layer` : class inherited from `Module`
            Слой из PyTorch
        `return_params_grad` : bool
            Если True, то вернуть еще градиенты параметров слоя
        Возвращает
        ----------
        `torch_layer_output` : torch.tensor
            Выход слоя `layer` после `forward()`
        `torch_layer_input_grad` : torch.tensor
            Градиент слоя `layer` после `backward()`
        `torch_params_grad` : torch.tensor [optional]
            Градиенты параметров слоя `layer`
        """

        layer_input.requires_grad = True
        torch_layer_output = torch_layer(layer_input)

        torch_layer_output.backward(next_layer_grad)
        torch_layer_input_grad = layer_input.grad
        if return_params_grad:
            torch_params_grad = torch_layer.parameters()
            return torch_layer_output.data, torch_layer_input_grad.data, torch_params_grad
        else:
            return torch_layer_output.data, torch_layer_input_grad.data

    @staticmethod
    def _train_torch_model(torch_model, data_train, n_epochs, loss, optimizer):
        """
        Функция для обучения модели. По-хорошему должна принимать ещё батч генератор, но пока без него.

        Параметры
        ---------
        `torch_model` : class object
            Модель для обучения.
        `data_train` : array like, shape=(1, 2)
            Набор данных для обучения. Содержит в себе 2 тензора: признаковые описания и целевую переменную.
        `n_epochs` : integer
            Число эпох обучения.
        `loss` : class object
            Функция риска.
        `optimizer` : class object
            Оптимизатор для параметров модели.
        """

        x_train, y_train = data_train
        x_train.requires_gradient = True
        y_train.requires_gradient = True

        for _ in range(n_epochs):
            # Обнуляем градиенты с предыдущей итерации
            torch_model.zero_grad()
            # Forward pass
            y_pred = torch_model.forward(x_train)
            loss = loss.forward(y_pred, y_train).unsqueeze(0).unsqueeze(0)
            # Backward pass
            loss.backward(y_pred, y_train)
            # Обновление весов
            optimizer.step()

    def test_FullyConnectedLayer(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in, n_out = 2, 3, 4

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.Linear(n_in, n_out)
            custom_layer = FullyConnectedLayer(n_in, n_out)
            custom_layer.W = torch_layer.weight.data.T
            custom_layer.b = torch_layer.bias.data

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in))
            next_layer_grad = self._generate_test_data((batch_size, n_out))

            # Тестируем наш слой
            result = self._custom_forward_backward(layer_input, next_layer_grad, custom_layer, return_params_grad=True)
            custom_layer_output, custom_layer_input_grad, custom_params_grad = result

            # Тестируем слой на PyTorch
            result = self._torch_forward_backward(layer_input, next_layer_grad, torch_layer, return_params_grad=True)
            torch_layer_output, torch_layer_input_grad, torch_params_grad = result

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6))
            # Сравниваем градиенты по входу слоя с точностью atol
            self.assertTrue(torch.allclose(torch_layer_input_grad, custom_layer_input_grad, atol=1e-6))
            # Сравниваем градиенты по параметрам слоя с точностью atol
            weight_grad, bias_grad = custom_params_grad
            torch_weight_grad = torch_layer.weight.grad.data
            torch_bias_grad = torch_layer.bias.grad.data
            self.assertTrue(torch.allclose(torch_weight_grad.T, weight_grad, atol=1e-6))
            self.assertTrue(torch.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_Softmax(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            custom_layer = Softmax()
            torch_layer = torch.nn.Softmax(dim=1)

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in))
            next_layer_grad = self._generate_test_data((batch_size, n_in))

            # Тестируем наш слой
            custom_layer_output, custom_layer_input_grad = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer
            )
            # Тестируем слой на PyTorch
            torch_layer_output, torch_layer_input_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer
            )

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(custom_layer_output, torch_layer_output, atol=1e-5))
            self.assertTrue(torch.allclose(custom_layer_input_grad, torch_layer_input_grad, atol=1e-5))

    def test_LogSoftmax(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            custom_layer = LogSoftmax()
            torch_layer = torch.nn.LogSoftmax(dim=1)

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in))
            next_layer_grad = self._generate_test_data((batch_size, n_in))

            # Тестируем наш слой
            custom_layer_output, custom_layer_input_grad = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer
            )
            # Тестируем слой на PyTorch
            torch_layer_output, torch_layer_input_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer
            )

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(custom_layer_output, torch_layer_output, atol=1e-5))
            self.assertTrue(torch.allclose(custom_layer_input_grad, torch_layer_input_grad, atol=1e-5))

    def test_FeedForwardModel(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in1, n_out1, n_out2 = 2, 3, 4, 5

        for _ in range(100):
            # Инициализируем слои
            # torch слои
            torch_model = torch.nn.Sequential(
                torch.nn.Linear(n_in1, n_out1),
                torch.nn.ReLU(),
                torch.nn.Linear(n_out1, n_out2),
                torch.nn.ReLU()
            )
            # Наши слои
            custom_model = FeedForwardModel([
                FullyConnectedLayer(n_in1, n_out1),
                ReLU(),
                FullyConnectedLayer(n_out1, n_out2),
                ReLU()
            ])
            # Сделаем начальные значения у всех параметров одинаковыми
            for custom_layer, torch_layer in zip(custom_model.layers, torch_model):
                if isinstance(torch_layer, type(torch.nn.Linear(1, 1))):
                    custom_layer.W = torch_layer.weight.data.T
                    custom_layer.b = torch_layer.bias.data

                # TODO: проверить, как хранятся параметры в torch, ибо то, что ниже, не работает как задумано
                # for custom_params, torch_params in zip(custom_layer.parameters, torch_layer.parameters()):
                #     if len(torch_params.shape) > 1:  # not bias case
                #         custom_params = torch_params.data.T
                #     else:  # bias case
                #         custom_params = torch_params.data

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in1), minval=-5, maxval=5)
            next_layer_grad = self._generate_test_data((batch_size, n_out2), minval=-5, maxval=5)

            # Тестируем наш слой
            result = self._custom_forward_backward(layer_input, next_layer_grad, custom_model, return_params_grad=True)
            custom_layer_output, custom_layer_input_grad, custom_params_grad = result
            # Тестируем слой на PyTorch
            result = self._torch_forward_backward(layer_input, next_layer_grad, torch_model, return_params_grad=True)
            torch_layer_output, torch_layer_input_grad, torch_params_grad = result

            # Сравниваем выходы и градиенты с точностью atol
            self.assertTrue(torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6))
            self.assertTrue(torch.allclose(torch_layer_input_grad, custom_layer_input_grad, atol=1e-4))

    def test_ReLU(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            custom_layer = ReLU()
            torch_layer = torch.nn.ReLU()

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in))
            next_layer_grad = self._generate_test_data((batch_size, n_in))

            # Тестируем наш слой
            custom_layer_output, custom_layer_input_grad = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer
            )
            # Тестируем слой на PyTorch
            torch_layer_output, torch_layer_input_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer
            )

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(custom_layer_output, torch_layer_output, atol=1e-6))
            self.assertTrue(torch.allclose(custom_layer_input_grad, torch_layer_input_grad, atol=1e-6))

    def test_Sigmoid(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            custom_layer = Sigmoid()
            torch_layer = torch.nn.Sigmoid()

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data((batch_size, n_in))
            next_layer_grad = self._generate_test_data((batch_size, n_in))

            # Тестируем наш слой
            custom_layer_output, custom_layer_input_grad = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer
            )
            # Тестируем слой на PyTorch
            torch_layer_output, torch_layer_input_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer
            )

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(custom_layer_output, torch_layer_output, atol=1e-6))
            self.assertTrue(torch.allclose(custom_layer_input_grad, torch_layer_input_grad, atol=1e-6))

    def test_MSE(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.MSELoss()
            custom_layer = MSE()

            # Формируем тестовые данные
            layer_input = self._generate_test_data((batch_size, n_in))
            target = torch.zeros((batch_size, n_in))

            # Тестируем прямой проход
            custom_layer_output = custom_layer.forward(layer_input, target)

            layer_input.requires_grad = True
            torch_layer_output = torch_layer.forward(layer_input, target)

            # Проверяем, что выходы близки
            self.assertTrue(torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6))

            # Тестируем обратный проход (градиенты)
            custom_layer_input_grad = custom_layer.backward(layer_input, target)
            torch_layer_output.backward()
            torch_layer_input_grad = layer_input.grad
            self.assertTrue(torch.allclose(torch_layer_input_grad, custom_layer_input_grad, atol=1e-6))

    def test_CrossEntropy(self):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.CrossEntropyLoss()
            custom_layer = CrossEntropy()

            # Формируем тестовые данные
            layer_input = self._generate_test_data((batch_size, n_in))
            target_labels = torch.from_numpy(np.random.choice(n_in, batch_size))

            # Тестируем прямой проход
            custom_layer_output = custom_layer.forward(layer_input, target_labels)

            layer_input.requires_grad = True
            torch_layer_output = torch_layer.forward(layer_input, target_labels)

            self.assertTrue(torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6))

            # Тестируем обратный проход (градиенты)
            custom_layer_input_grad = custom_layer.backward(layer_input, target_labels)
            torch_layer_output.backward()
            torch_layer_grad_var = layer_input.grad
            self.assertTrue(torch.allclose(torch_layer_grad_var, custom_layer_input_grad, atol=1e-6))

    def test_KLDivergence(self):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        batch_size, n_in = 2, 4

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.KLDivLoss()
            custom_layer = KLDivergence()

            # Формируем тестовые данные
            layer_input = self._generate_test_data((batch_size, n_in))
            layer_input.requires_grad = False
            layer_input = torch.nn.LogSoftmax(dim=1).forward(layer_input)

            target_labels = np.random.choice(n_in, batch_size)
            target = torch.zeros((batch_size, n_in))
            target[np.arange(batch_size), target_labels] = 1  # one-hot encoding

            # Тестируем прямой проход
            custom_layer_output = custom_layer.forward(layer_input, target)

            layer_input.requires_grad = True
            torch_layer_output = torch_layer(layer_input, target)

            self.assertTrue(torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6))

            # Тестируем обратный проход
            custom_layer_grad = custom_layer.backward(layer_input, target)
            torch_layer_output.backward()
            torch_layer_grad = torch_layer_output.grad

            self.assertTrue(torch.allclose(torch_layer_grad, custom_layer_grad, atol=1e-6))

    def test_MSE_with_regularization(self):
        raise NotImplementedError
    #     torch.manual_seed(RANDOM_SEED)
    #     batch_size, n_in, n_out = 2, 3, 4
    #     l1, l2 = 0.1, 10.
    #
    #     for _ in range(100):
    #         # Формируем тестовые данные
    #         layer_input = self._generate_test_data((batch_size, n_in))
    #         target = torch.zeros((batch_size, n_in))
    #         params = self._generate_test_data((n_in, n_out))
    #
    #         # Инициализируем слои
    #         torch_layer = torch.nn.MSELoss()
    #         custom_layer_l1 = MSE(l1=l1, params=params)
    #         custom_layer_l2 = MSE(l2=l2, params=params)
    #
    #         # Тестируем прямой проход
    #         custom_layer_output_l1 = custom_layer_l1.forward(layer_input, target)
    #         custom_layer_output_l2 = custom_layer_l2.forward(layer_input, target)
    #
    #         layer_input.requires_grad = True
    #         torch_layer_output = torch_layer.forward(layer_input, target)
    #         l1_penalty = params.abs().sum()
    #         l2_penalty = params.square().sum()
    #         loss_l1 = torch_layer_output + l1 * l1_penalty
    #         loss_l2 = torch_layer_output + l2 * l2_penalty
    #
    #         # Проверям, что выходы близки
    #         self.assertTrue(torch.allclose(loss_l1, custom_layer_output_l1, atol=1e-6))
    #         self.assertTrue(torch.allclose(loss_l2, custom_layer_output_l2, atol=1e-6))
    #
    #         # Тестируем обратный проход (градиенты)
    #         custom_layer_input_grad_l1 = custom_layer_l1.backward(layer_input, target)
    #         custom_layer_input_grad_l2 = custom_layer_l2.backward(layer_input, target)
    #
    #         loss_l1.backward()
    #         torch_layer_input_grad_l1 = layer_input.grad
    #         self.assertTrue(torch.allclose(torch_layer_input_grad_l1, custom_layer_input_grad_l1, atol=1e-6))
    #         layer_input.zero_grad()
    #         loss_l2.backward()
    #         torch_layer_input_grad_l2 = layer_input.grad
    #         self.assertTrue(torch.allclose(torch_layer_input_grad_l2, custom_layer_input_grad_l2, atol=1e-6))

    def test_SimpleSGD(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, n_in, n_out = 1, 1, 1

        # Формируем тестовые данные
        model_input = self._generate_test_data((batch_size, n_in))
        target = self._generate_test_data((batch_size, n_out))

        # Задаём модели
        torch_model = torch.nn.Sequential(torch.nn.Linear(n_in, n_out, bias=False))
        # TODO: как получше сделать получение весов для модели из торча?
        weight_init = None
        for layer in torch_model:
            weight_init = layer.weight.data.T

        custom_model = FeedForwardModel(
            layers=[
                FullyConnectedLayer(
                    n_in, n_out, bias=False,
                    init=weight_init  # Так как слой всего один, это то, что нужно
                )
            ],
            loss=MSE(),
            optimizer=GradientDescend(lr=0.01)
        )

        # Обучаем
        custom_model.train([model_input, target], n_epochs=20)
        self._train_torch_model(torch_model, [model_input, target], 20, torch.nn.MSELoss(),
                                torch.optim.SGD(torch_model.parameters(), lr=0.01)
                                )

        # Сравниваем параметры
        torch_weight = None
        for layer in torch_model:
            torch_weight = layer.weight.data.T
        self.assertTrue(torch.allclose(custom_model.parameters, torch_weight))

    def test_MomentumSGD(self):
        raise NotImplementedError

    def test_NesterovSGD(self):
        raise NotImplementedError

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)
