import argparse
import copy
import json
import math
import os
from typing import Type

import torch
from torch.optim.optimizer import Optimizer
from torchvision.datasets import VisionDataset

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.model_base import BaseModel
from nn.utils.is_using_cuda import is_using_cuda
from nn.utils.summary import summary
from runs.r_2021_08_31._run import RUN_MODELS
from utils.data_dirs import data_dirs, erase_data_dirs
from utils.path_functions import get_relative_path, path_join

DATA_FOLDER = "C:/_data"
VERBOSE_LOG_FILE = True
SAVE_ALL_MODEL = False
DRY_RUN = False


def main(
        name: str,
        dataset: Type[VisionDataset],
        batch_size: int,
        epochs: int,
        model_class: Type[BaseModel],
        optimizer: Type[Optimizer],
        loss_fn,
        normalizer_fn,
        model_kargs
):
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

    model = model_class(
        in_features=tuple(input_shape[1:]),
        out_features=len(classes),
        **model_kargs
    )
    model.optimizer = optimizer(model.parameters())
    model.loss = loss_fn
    model.create_tensorboard(log_dir=tensorboard_path)
    model.compile(device)

    model.tensorboard.tensorboard.add_text("dataset", str(dataset))
    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = str(model.optimizer)
            kwargs["loss_fn"] = str(model.loss)
            kwargs["dataset"] = str(dataset)
            file.write(json.dumps(kwargs, sort_keys=True, indent=2) + "\n\n")
            file.write(str(model) + "\n\n")
            file.write(summary(model, input_size=tuple(input_shape[1:])) + "\n\n")

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

        if SAVE_ALL_MODEL:
            torch.save(model.state_dict(),
                       path_join(models_path, f"{name_with_timestamp}_{str_epoch}_{dataset.__name__}.pt"))

        if normalizer_fn is not None:
            normalizer_fn(model)

        if DRY_RUN:
            break

    model.tensorboard.tensorboard.add_hparams(
        hparam_dict={
            "model": model_class.__name__,
            "dataset": dataset.__name__,
            "batch_size": batch_size,
            "optimizer": optimizer.__name__,
            "approach": model.approach.value,
            "std": -1 if model.std is None else model.std,
            "activation_class": model.activation_class.__name__,
            "normalize_fn": "None" if normalizer_fn is None else normalizer_fn.__name__,
        },
        metric_dict={
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        run_name=model_class.__name__,
    )


def only_run_this_main(prefix):
    # global DATA_FOLDER
    # DATA_FOLDER = f"C:/_data_{prefix}"
    erase_data_dirs(DATA_FOLDER)
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    for i, name in enumerate(sorted(list(RUN_MODELS.keys()))):
        if name.startswith(prefix):
            torch.manual_seed(0)
            parameters = copy.deepcopy(RUN_MODELS[name])
            parameters["name"] = name
            main(**parameters)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', type=str, required=True)
    # args = parser.parse_args()
    # only_run_this_main(args.p)
    only_run_this_main("")
