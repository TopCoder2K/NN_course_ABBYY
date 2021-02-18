import torch
from nn.data import get_line


def main():
    torch.manual_seed(1)
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.5)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    x, y = get_line()
    n_epochs = 20
    for _ in range(n_epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
