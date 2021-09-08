import torch
from torch.utils.data import DataLoader


def test(model, test_loader: DataLoader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            output = model.output(data)
            loss, accuracy = model.loss(output, target)
            total_loss += loss.item()
            correct += accuracy

    total_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return total_loss, accuracy
