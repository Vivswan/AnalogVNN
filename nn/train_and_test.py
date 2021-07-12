import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(model, device: torch.device, train_loader: DataLoader, optimizer: Optimizer, loss_fn, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
        loss, item_loss, item_correct = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += item_loss
        correct += item_correct

        print_mod = int(len(train_loader.dataset) / (len(data) * 5))
        if batch_idx % print_mod == 0 and batch_idx > 0:
            print(
                f'Train Epoch:'
                f' {((epoch + 1) if epoch is not None else "")}'
                f' '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)}'
                f' ({100. * batch_idx / len(train_loader):.0f}%)'
                f']'
                f'\tLoss: {total_loss / (batch_idx * len(data)):.6f}'
            )

    total_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return total_loss, accuracy


def test(model, device: torch.device, test_loader: DataLoader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss, item_loss, item_correct = loss_fn(output, target)
            total_loss += item_loss
            correct += item_correct

    total_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return total_loss, accuracy
