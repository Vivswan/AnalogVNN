import torch
import torch.nn as nn

nllloss_fn = nn.NLLLoss(reduction='sum')
criterion = nn.CrossEntropyLoss()


def nll_loss_fn(output, target):
    loss = nllloss_fn(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, loss.item(), correct


def cross_entropy_loss_fn(output, target):
    loss = criterion(output, target)
    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return loss, loss.item() * target.shape[0], correct
