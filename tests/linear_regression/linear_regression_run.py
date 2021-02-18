import torch

from nn.layers import FullyConnectedLayer
from nn.models import FeedForwardModel
from nn.losses import MSE
from nn.optimizer import GradientDescend
from nn.data import get_line


def main():
    torch.manual_seed(1)
    data = get_line(seed=1)

    model = FeedForwardModel(
        layers=[
            FullyConnectedLayer(
                1, 1, bias=False,
                init=torch.full((1, 1), 0.5, dtype=torch.float)
            )
        ],
        loss=MSE(),
        optimizer=GradientDescend(lr=0.3)
    )
    model.train(data, n_epochs=20)


if __name__ == '__main__':
    with torch.no_grad():
        main()
