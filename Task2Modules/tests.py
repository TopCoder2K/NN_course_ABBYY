import torch
import unittest

from CNN import Conv, AvgPool
from OtherLayers import BatchNorm2d

RANDOM_SEED = 1


class TestLayers(unittest.TestCase):
    """Класс для тестирования целевых для этого задания модулей."""

    @staticmethod
    def _generate_test_data(shape, minval=-10, maxval=10):
        """Генерирует тестовые данные для `forward()` и `backward()`."""

        rand_array = torch.distributions.uniform.Uniform(
            minval, maxval
        ).sample(shape)
        # Иногда имеет смысл нормализовать
        # rand_array /= rand_array.sum(axis=-1, keepdims=True)
        # rand_array = rand_array.clip(1e-5, 1.)
        # rand_array = 1. / rand_array
        return rand_array

    @staticmethod
    def _custom_forward_backward(layer_input, next_layer_grad, custom_layer,
                                 returns_params_grad=False):
        """
        Вычисляет результат `forward()` и `backward()` в слое `layer` для
        небиблиотечной реализации.

        Параметры
        ---------
        `layer_input` : torch.tensor
            Тестовый вход.
        `next_layer_grad` : torch.tensor
            Тестовый градиент, пришедший от следующего слоя.
        `layer` : class object
            Слой из нашего мини-фреймворка.
        `return_params_grad` : bool
            Возвращает ли переданный слой градиенты по параметрам? Если True,
            выход `backward()` будет распакован на dX, dW, db.

        Возвращает
        ----------
        `custom_layer_output` : torch.tensor
            Выход слоя `layer` после `forward()`.
        `dX` : torch.tensor
            Градиент лосса по входу слоя.
        `dW` : torch.tensor [optional]
            Градиент лосса по весам слоя.
        `db` : torch.tensor [optional]
            Градиент лосса по параметрам сдвига слоя.
        """

        custom_layer_output = custom_layer.forward(layer_input)
        custom_layer_input_grad = custom_layer.backward(next_layer_grad)
        if returns_params_grad:
            dX, dW, db = custom_layer_input_grad
            return custom_layer_output, dX, dW, db
        else:
            return custom_layer_output, custom_layer_input_grad

    @staticmethod
    def _torch_forward_backward(layer_input, next_layer_grad, torch_layer,
                                return_params_grad=False):
        """
        Вычисляет результат `forward()` и `backward()` в PyTorch-слое `layer`.

        Параметры
        ---------
        `layer_input` : torch.tensor
            Тестовый вход.
        `next_layer_grad` : torch.tensor
            Тестовый градиент, пришедший от следующего слоя.
        `torch_layer` : class inherited from `Module`
            Слой из PyTorch.
        `return_params_grad` : bool
            Если True, то вернуть еще градиенты параметров слоя.

        Возвращает
        ----------
        `torch_layer_output` : torch.tensor
            Выход слоя `layer` после `forward()`.
        `torch_layer_input_grad` : torch.tensor
            Градиент слоя `layer` после `backward()`.
        `torch_params_grad` : torch.tensor [optional]
            Градиенты параметров слоя `layer`.
        """

        layer_input.requires_grad = True
        torch_layer_output = torch_layer(layer_input)

        torch_layer_output.backward(next_layer_grad)
        torch_layer_input_grad = layer_input.grad
        if return_params_grad:
            torch_params_grad = torch_layer.parameters()
            return torch_layer_output.data, torch_layer_input_grad.data, \
                   torch_params_grad
        else:
            return torch_layer_output.data, torch_layer_input_grad.data

    def test_Conv(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, img_size, filter_size, in_chs, out_chs = 1, 5, 5, 1, 4
        padding, stride = 0, 1

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.Conv2d(in_chs, out_chs, filter_size,
                                          padding=padding, stride=stride)
            custom_layer = Conv(nb_filters=in_chs, nb_channels=out_chs,
                                filter_size=filter_size, stride=stride,
                                padding=padding)
            custom_layer.W = torch_layer.weight.detach().clone()
            custom_layer.b = torch_layer.bias.detach().clone()

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data(
                (batch_size, in_chs, img_size, img_size)
            )
            new_img_size = int(
                (img_size + 2 * padding - filter_size) / stride
            ) + 1
            next_layer_grad = self._generate_test_data(
                (batch_size, out_chs, new_img_size, new_img_size)
            )

            # Тестируем наш слой
            custom_layer_output, dX, dW, db = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer,
                returns_params_grad=True
            )
            # Тестируем слой на PyTorch
            torch_result = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer,
                return_params_grad=True
            )
            torch_output, torch_input_grad, torch_params_grad = torch_result

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(
                torch_output, custom_layer_output, atol=1e-6
            ))
            # Сравниваем градиенты по входу слоя с точностью atol
            self.assertTrue(torch.allclose(torch_input_grad, dX, atol=1e-6))
            # Сравниваем градиенты по параметрам слоя с точностью atol
            torch_weight_grad = torch_layer.weight.grad.data
            torch_bias_grad = torch_layer.bias.grad.data
            self.assertTrue(torch.allclose(torch_bias_grad, db, atol=1e-6))
            self.assertTrue(torch.allclose(torch_weight_grad, dW, atol=1e-6))

    def test_AvgPool(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, img_size, filter_size, in_chs = 1, 8, 3, 1
        padding, stride = 0, 1

        for _ in range(100):
            # Инициализируем слои
            torch_layer = torch.nn.AvgPool2d(filter_size, stride, padding)
            custom_layer = AvgPool(filter_size, stride, padding)

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data(
                (batch_size, in_chs, img_size, img_size)
            )
            new_img_size = int(
                1 + (img_size + 2 * padding - filter_size) / stride
            )
            next_layer_grad = self._generate_test_data(
                (batch_size, in_chs, new_img_size, new_img_size)
            )

            # Тестируем наш слой
            custom_layer_output, dX = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer,
                returns_params_grad=False
            )
            # Тестируем слой на PyTorch
            torch_output, torch_input_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer,
                return_params_grad=False
            )

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(
                torch_output, custom_layer_output, atol=1e-6
            ))
            # Сравниваем градиенты по входу слоя с точностью atol
            self.assertTrue(torch.allclose(torch_input_grad, dX, atol=1e-6))

    def test_BatchNorm2d(self):
        torch.manual_seed(RANDOM_SEED)
        batch_size, nb_channels, h, w = 32, 16, 5, 5

        for _ in range(100):
            # Инициализируем слои
            alpha = 0.9
            # По умолчанию training == True
            custom_layer = BatchNorm2d(nb_channels, alpha)
            torch_layer = torch.nn.BatchNorm2d(
                num_features=nb_channels, eps=custom_layer.eps,
                momentum=1.-alpha, affine=True
            )
            custom_layer.moving_mean = torch_layer.running_mean.detach().clone()
            custom_layer.moving_var = torch_layer.running_var.detach().clone()

            # Формируем тестовые входные тензоры
            layer_input = self._generate_test_data(
                (batch_size, nb_channels, h, w)
            )
            next_layer_grad = self._generate_test_data(
                (batch_size, nb_channels, h, w)
            )

            # Тестируем наш слой
            custom_layer_output, dX, dW, db = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer,
                returns_params_grad=True
            )
            # Тестируем слой на PyTorch
            torch_result = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer,
                return_params_grad=True
            )
            torch_output, torch_input_grad, torch_params_grad = torch_result

            # Сравниваем выходы с точностью atol
            self.assertTrue(torch.allclose(
                torch_output, custom_layer_output, atol=1e-6
            ))
            # Сравниваем градиенты по входу
            self.assertTrue(torch.allclose(torch_input_grad, dX, atol=1e-6))
            # Сравниваем градиенты по параметрам
            torch_weight_grad = torch_layer.weight.grad.data
            torch_bias_grad = torch_layer.bias.grad.data
            self.assertTrue(torch.allclose(torch_bias_grad, db, atol=1e-6))
            self.assertTrue(torch.allclose(torch_weight_grad, dW, atol=1e-6))

            # Сравниваем moving mean и moving variance
            self.assertTrue(torch.allclose(
                custom_layer.moving_mean, torch_layer.running_mean.numpy()
            ))
            # мы не проверяем moving variance, потому что в PyTorch используется
            # немного другая формула: var * N / (N-1) (несмещенная оценка)
            self.assertTrue(torch.allclose(
                custom_layer.moving_var, torch_layer.running_var
            ))

            # Тестируем BatchNorm-ы на стадии evaluation
            custom_layer.moving_var = torch_layer.running_var.detach().clone()
            custom_layer.eval()
            torch_layer.eval()

            custom_layer_output, dX, dW, db = self._custom_forward_backward(
                layer_input, next_layer_grad, custom_layer,
                returns_params_grad=True
            )
            torch_layer_output, torch_layer_grad = self._torch_forward_backward(
                layer_input, next_layer_grad, torch_layer,
                return_params_grad=False
            )

            self.assertTrue(torch.allclose(
                torch_layer_output, custom_layer_output, atol=1e-6
            ))
