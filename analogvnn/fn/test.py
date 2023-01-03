import torch
from torch.utils.data import DataLoader


def test(model, test_loader: DataLoader, test_run=False):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_size = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)

            output = model(data)
            loss, accuracy = model.loss(output, target)

            total_loss += loss.item() * len(data)
            total_accuracy += accuracy * len(data)
            total_size += len(data)

            if test_run:
                break

    total_loss /= total_size
    total_accuracy /= total_size
    return total_loss, total_accuracy
