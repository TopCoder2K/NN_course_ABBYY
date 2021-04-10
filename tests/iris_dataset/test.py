import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from nn.layers import FullyConnectedLayer
from nn.models import FeedForwardModel
from nn.losses import CrossEntropy
from nn.activations import Sigmoid, ReLU
from nn.optimizer import GradientDescend
from nn.data import get_iris

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('notebook', font_scale=1.7)


def main():
    torch.manual_seed(1)
    x, y = get_iris()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

    with torch.no_grad():
        custom_model = FeedForwardModel(
            layers=[
                FullyConnectedLayer(4, 30),
                Sigmoid(),
                FullyConnectedLayer(30, 100),
                Sigmoid(),
                FullyConnectedLayer(100, 30),
                Sigmoid(),
                FullyConnectedLayer(30, 3)
            ],
            loss=CrossEntropy(),
            optimizer=GradientDescend(lr=0.3, momentum=0.5)
        )
        loss_history = custom_model.train(data_train=[x_train, y_train.squeeze()], n_epochs=500)

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title('Зависимость MSE() от эпохи (с momentum=0.5)')
        plt.show()
        plt.savefig('WithMomentum.png')
        print(f'Custom accuracy: {accuracy_score(custom_model.forward(x_test).numpy().argmax(axis=1), y_test): .4f}')

    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(x_train, y_train.squeeze())

    print(f'sklearn `LogisticRegression` accuracy: {accuracy_score(log_reg.predict(x_test), y_test): .4f}')
    print('=================================================================')


if __name__ == '__main__':
    with torch.no_grad():
        main()
