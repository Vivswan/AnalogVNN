import json
import math
from typing import Type

import torch
from torch.optim.optimizer import Optimizer
from torchvision.datasets import VisionDataset

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.model_base import BaseModel
from nn.utils.summary import summary
from runs.r_2021_07_20.run import RUN_MODELS_20210720
from utils.data_dirs import data_dirs, erase_data_dirs
from nn.utils.is_using_cuda import is_using_cuda
from utils.path_functions import get_relative_path, path_join

torch.manual_seed(0)

DATA_FOLDER = get_relative_path(__file__, "data")
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
    kwargs = {}

    name_with_timestamp, models_path, tensorboard_path, dataset_path = data_dirs(DATA_FOLDER, name=name)
    device, is_cuda = is_using_cuda()
    log_file = path_join(DATA_FOLDER, f"{name_with_timestamp}_logs.txt")

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=dataset,
        path=dataset_path,
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    nn = model(
        in_features=input_shape,
        out_features=len(classes),
        device=device,
        **model_kargs
    )
    nn.compile(
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    print(device)

    nn.tb.add_text("dataset", str(dataset))
    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = str(nn.optimizer)
            kwargs["loss_fn"] = str(nn.loss)
            kwargs["dataset"] = str(dataset)
            file.write(json.dumps(kwargs, sort_keys=True, indent=2) + "\n\n")
            file.write(str(nn) + "\n\n")
            file.write(summary(nn) + "\n\n")

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

    nn.close()


if __name__ == '__main__':
    erase_data_dirs(DATA_FOLDER)
    for i in RUN_MODELS_20210720:
        # i = "ReducePrecision-16_GaussianNoise-0.05"
        print(i)
        parameters = {
            **RUN_MODELS_20210720[i],
            "name": i,
        }
        main(**parameters)
