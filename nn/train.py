from torch.utils.data import DataLoader


def train(model, train_loader: DataLoader, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    dataset_size = len(list(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model.device), target.to(model.device)

        # zero the parameter gradients
        model.zero_grad()
        model.optimizer.zero_grad()

        # forward + backward + optimize
        output = model.output(data)
        loss, accuracy = model.loss(output, target)

        model.backward()
        model.optimizer.step()

        # print statistics
        total_loss += loss.item()
        correct += accuracy

        print_mod = int(dataset_size / (len(data) * 5))
        if print_mod > 0 and batch_idx % print_mod == 0 and batch_idx > 0:
            print(
                f'Train Epoch:'
                f' {((epoch + 1) if epoch is not None else "")}'
                f' '
                f'[{batch_idx * len(data)}/{dataset_size}'
                f' ({100. * batch_idx / len(train_loader):.0f}%)'
                f']'
                f'\tLoss: {total_loss / (batch_idx * len(data)):.6f}'
            )

    total_loss /= dataset_size
    accuracy = correct / dataset_size
    return total_loss, accuracy
