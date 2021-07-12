import math

import torchvision.datasets
from torch import optim

from dataloaders.load_mnist import load_dataset
from dataloaders.loss_functions import nll_loss_fn
from models.a_single_linear_layer_model import SingleLinearLayerModel
from nn.summary import summary
from utils.data_dirs import data_dirs
from utils.is_using_cuda import is_using_cuda
from utils.path_functions import get_relative_path, path_join

DATA_FOLDER = get_relative_path(__file__, "D:/_data")
VERBOSE_LOG_FILE = True


def main(
        epochs=10,
        dry_run=False,
        kwargs=None,
):
    if kwargs is None:
        kwargs = {}

    name, models_path, tensorboard_path = data_dirs(DATA_FOLDER, name="SingleLinearLayerModel")
    device, is_cuda = is_using_cuda()
    log_file = path_join(DATA_FOLDER, f"{name}_logs.txt")

    train_loader, test_loader, input_shape, classes = load_dataset(
        dataset=torchvision.datasets.MNIST,
        path=path_join(DATA_FOLDER, "datasets"),
        batch_size=100,
        is_cuda=is_cuda
    )

    model = SingleLinearLayerModel(
        in_features=input_shape,
        out_features=len(classes),
        device=device,
        log_dir=tensorboard_path,
    )
    model.compile(
        optimizer=optim.Adam,
        loss_fn=nll_loss_fn
    )

    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = model.optimizer
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
        if VERBOSE_LOG_FILE:
            with open(log_file, "a+") as file:
                file.write(print_str)

        if dry_run:
            break

    model.close()


if __name__ == '__main__':
    main(dry_run=True)
