import math

import torch
import torchvision.datasets
from torch import optim

from dataloaders.load_mnist import load_dataset
from dataloaders.loss_functions import nll_loss_fn
from models.b_reduce_precision_layer_model import ReducePrecisionLayerModel
from nn.summary import summary
from utils.data_dirs import data_dirs
from utils.is_using_cuda import is_using_cuda
from utils.path_functions import get_relative_path, path_join


def main():
    data_folder = get_relative_path(__file__, "D:/_data")
    verbose_log_file = True
    save_all_model = False

    # model_class = SingleLinearLayerModel
    model_class = ReducePrecisionLayerModel
    dataset = torchvision.datasets.MNIST
    epochs = 10

    name = model_class.__name__
    dry_run = False
    kwargs = {}

    name_with_timestamp, models_path, tensorboard_path, dataset_path = data_dirs(data_folder, name=name)
    device, is_cuda = is_using_cuda()
    log_file = path_join(data_folder, f"{name_with_timestamp}_logs.txt")

    train_loader, test_loader, input_shape, classes = load_dataset(
        dataset=dataset,
        path=dataset_path,
        batch_size=1000,
        is_cuda=is_cuda
    )

    model = model_class(
        in_features=input_shape,
        out_features=len(classes),
        device=device,
        log_dir=tensorboard_path,
    )
    model.compile(
        optimizer=optim.Adam,
        loss_fn=nll_loss_fn
    )

    if verbose_log_file:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = model.optimizer
            kwargs["loss_fn"] = model.loss_fn
            file.write(str(kwargs) + "\n\n")
            file.write(str(model) + "\n\n")
            file.write(summary(model) + "\n\n")

    for epoch in range(epochs):
        train_loss, train_accuracy, test_loss, test_accuracy = model.fit(train_loader, test_loader, epoch)

        str_epoch = str(epoch).zfill(math.ceil(math.log10(epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'

        print(print_str)
        if verbose_log_file:
            with open(log_file, "a+") as file:
                file.write(print_str)

        if save_all_model:
            torch.save(model.state_dict(),
                       path_join(models_path, f"{name_with_timestamp}_{str_epoch}_{dataset.__name__}.pt"))

        if dry_run:
            break

    model.close()


if __name__ == '__main__':
    main()
