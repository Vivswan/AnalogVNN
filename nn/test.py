import torch
from torch.utils.data import DataLoader


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
