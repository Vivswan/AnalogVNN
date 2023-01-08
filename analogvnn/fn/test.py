from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

__all__ = ['test']


def test(model: nn.Module, test_loader: DataLoader, test_run: bool = False) -> Tuple[float, float]:
    """Test the model on the test set.

    Args:
        model (torch.nn.Module): the model to test.
        test_loader (DataLoader): the test set.
        test_run (bool): is it a test run.

    Returns:
        tuple: the loss and accuracy of the model on the test set.
    """

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
