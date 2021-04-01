import torch
import numpy as np


def main():
    torch_layer_output = model.output

    assert np.allclose(torch_layer_output, custom_layer_output, atol=1e-6)
    assert np.allclose(torch_layer_grad, custom_layer_grad, atol=1e-4)
    weight_grad, bias_grad = custom_params_grad[1]
    torch_weight_grad = torch_layer.weight.grad.data.numpy()
    torch_bias_grad = torch_layer.bias.grad.data.numpy()
    self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
    self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))


if __name__ == '__main__':
    with torch.no_grad():
        main()
