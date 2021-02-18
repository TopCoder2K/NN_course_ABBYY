import torch
import numpy as np, numpy.random as nr
# import matplotlib.pyplot as plt


def get_line(line=None, n_points=64, sigma=0.05, seed=1):
    if line is None:  # default line: y = \pi * x + 0
        line = np.asarray([-np.pi, 1, 0])

    nr.seed(seed)
    points = np.zeros((2, n_points), dtype=np.float32)
    points[0] = nr.rand(n_points)  # x in [0, 1]
    points[1] = -(line[0] * points[0] + line[2]) / line[1]  # line
    points += nr.randn(2, n_points) * sigma  # Gaussian noise
    # plt.plot(points[0], points[1], '.')
    return torch.tensor(points, dtype=torch.float).T.split([1, 1], dim=1)


def get_iris():
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)
    x = torch.tensor(x, dtype=torch.float)
    # x = (x - x.mean()) / x.std()
    y = torch.tensor(np.expand_dims(y, 1), dtype=torch.float)
    return x, y
