from torch.utils.data import DataLoader


def train(
        model,
        train_loader: DataLoader,
        epoch=None,
        test_run=False
):
    model.train()
    total_loss = 0.0
    total_accuracy = 0
    total_size = 0
    if isinstance(train_loader, DataLoader):
        # noinspection PyTypeChecker
        dataset_size = len(train_loader.dataset)
    else:
        dataset_size = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model.device), target.to(model.device)

        # zero the parameter gradients
        model.zero_grad()
        model.optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
        loss, accuracy = model.loss(output, target)

        model.backward()
        model.optimizer.step()

        # print statistics
        total_loss += loss.item() * len(data)
        total_accuracy += accuracy * len(data)
        total_size += len(data)

        print_mod = int(dataset_size / (len(data) * 5))
        if print_mod > 0 and batch_idx % print_mod == 0 and batch_idx > 0:
            print(
                f'Train Epoch:'
                f' {((epoch + 1) if epoch is not None else "")}'
                f' [{batch_idx * len(data)}/{dataset_size} ({100. * batch_idx / len(train_loader):.0f}%)]'
                f'\tLoss: {total_loss / total_size:.6f}'
                f'\tAccuracy: {total_accuracy / total_size * 100:.2f}%'
            )

        if test_run:
            break

    total_loss /= total_size
    total_accuracy /= total_size
    return total_loss, total_accuracy
