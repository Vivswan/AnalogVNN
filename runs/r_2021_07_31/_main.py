import copy
import json
import math
from typing import Type

import torch
from torch.optim.optimizer import Optimizer
from torchvision.datasets import VisionDataset

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.model_base import BaseModel
from nn.utils.is_using_cuda import is_using_cuda
from nn.utils.summary import summary
from runs.r_2021_07_31._run import RUN_MODELS
from utils.data_dirs import data_dirs, erase_data_dirs
from utils.path_functions import get_relative_path, path_join

DATA_FOLDER = get_relative_path(__file__, "D:/_data")
VERBOSE_LOG_FILE = True
SAVE_ALL_MODEL = False
DRY_RUN = False


def main(
        name: str,
        dataset: Type[VisionDataset],
        batch_size: int,
        epochs: int,
        model: Type[BaseModel],
        optimizer: Type[Optimizer],
        loss_fn,
        model_kargs
):
    torch.manual_seed(0)
    kwargs = {}

    name_with_timestamp, models_path, tensorboard_path, dataset_path = data_dirs(DATA_FOLDER, name=name)
    device, is_cuda = is_using_cuda()
    log_file = path_join(DATA_FOLDER, f"{name_with_timestamp}_logs.txt")
    print(f"{name}, {device}")

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    nn = model(
        in_features=tuple(input_shape[1:]),
        out_features=len(classes),
        **model_kargs
    )
    nn.optimizer = optimizer(nn.parameters())
    nn.loss = loss_fn
    nn.create_tensorboard(log_dir=tensorboard_path)
    nn.compile(device)

    nn.tensorboard.tensorboard.add_text("dataset", str(dataset))
    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = str(nn.optimizer)
            kwargs["loss_fn"] = str(nn.loss)
            kwargs["dataset"] = str(dataset)
            file.write(json.dumps(kwargs, sort_keys=True, indent=2) + "\n\n")
            file.write(str(nn) + "\n\n")
            file.write(summary(nn, input_size=tuple(input_shape[1:])) + "\n\n")

    for epoch in range(epochs):
        train_loss, train_accuracy, test_loss, test_accuracy = nn.fit(train_loader, test_loader, epoch)

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

        if SAVE_ALL_MODEL:
            torch.save(nn.state_dict(),
                       path_join(models_path, f"{name_with_timestamp}_{str_epoch}_{dataset.__name__}.pt"))

        if DRY_RUN:
            break


if __name__ == '__main__':
    erase_data_dirs(DATA_FOLDER)
    for i in RUN_MODELS:
        torch.manual_seed(0)
        parameters = copy.deepcopy(RUN_MODELS[i])
        parameters["name"] = i
        main(**parameters)
