import torch

from nn.layers import FullyConnectedLayer
from nn.models import FeedForwardModel
from nn.losses import MSE
from nn.optimizer import GradientDescend
from nn.data import get_line


def simple_regression_test():
    torch.manual_seed(1)
    x, y = get_line()

    with torch.no_grad():
        custom_model = FeedForwardModel(
            layers=[
                FullyConnectedLayer(
                    1, 1, bias=False,
                    init=torch.full((1, 1), 0.5, dtype=torch.float)
                )
            ],
            loss=MSE(),
            optimizer=GradientDescend(lr=0.3)
        )
        custom_model.train(data_train=[x, y], n_epochs=20)

        print('Custom model parameters:\n', custom_model.parameters)

    torch_model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(torch_model.weight, 0.5)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.3)

    n_epochs = 20
    for _ in range(n_epochs):
        pred = torch_model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Torch model parameters:')
    for param in torch_model.parameters():
        print(param)
    print('=================================================================')


if __name__ == '__main__':
    simple_regression_test()
