import json
import math
from typing import Type

import torch
import torchvision.datasets
from torch import optim
from torch.optim.optimizer import Optimizer
from torchvision.datasets import VisionDataset

from dataloaders.load_vision_dataset import load_vision_dataset
from dataloaders.loss_functions import nll_loss_fn
from models.a_single_linear_layer_model import SingleLinearLayerModel
from models.b_reduce_precision_layer_model import ReducePrecisionLayerModel
from models.c_gaussian_reduce_precision_layer_model import GaussianNoiseReducePrecisionLayerModel
from nn.model_base import ModelBase
from nn.summary import summary
from utils.data_dirs import data_dirs
from utils.is_using_cuda import is_using_cuda
from utils.path_functions import get_relative_path, path_join

torch.manual_seed(0)

DEFAULT_PARAMETERS = {
    "dataset": torchvision.datasets.MNIST,
    "batch_size": 10,
    "epochs": 10,
    "optimizer": optim.Adam,
    "loss_fn": nll_loss_fn,
    "model_kargs": {},
}

TEST_MODELS = {
    "SingleLinear": {
        **DEFAULT_PARAMETERS,
        "model": SingleLinearLayerModel
    },
    "ReducePrecision-2": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=2)
    },
    "ReducePrecision-4": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=4)
    },
    "ReducePrecision-8": {
        **DEFAULT_PARAMETERS,
        "model": ReducePrecisionLayerModel,
        "model_kargs": dict(precision=8)
    },
    "ReducePrecision-2_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=2, std=0.1)
    },
    "ReducePrecision-2_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=2, std=0.05)
    },
    "ReducePrecision-4_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=4, std=0.1)
    },
    "ReducePrecision-4_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=4, std=0.05)
    },
    "ReducePrecision-8_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=8, std=0.1)
    },
    "ReducePrecision-8_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=8, std=0.05)
    },
    "ReducePrecision-16_GaussianNoise-0.1": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=16, std=0.1)
    },
    "ReducePrecision-16_GaussianNoise-0.05": {
        **DEFAULT_PARAMETERS,
        "model": GaussianNoiseReducePrecisionLayerModel,
        "model_kargs": dict(precision=16, std=0.05)
    },
}

DATA_FOLDER = get_relative_path(__file__, "D:/_data")
VERBOSE_LOG_FILE = True
SAVE_ALL_MODEL = False
DRY_RUN = False


def main(
        name: str,
        dataset: Type[VisionDataset],
        batch_size: int,
        epochs: int,
        model: Type[ModelBase],
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
        log_dir=tensorboard_path,
        **model_kargs
    )
    nn.compile(
        optimizer=optimizer,
        loss_fn=loss_fn
    )

    nn.tb.add_text("dataset", str(dataset))
    if VERBOSE_LOG_FILE:
        with open(log_file, "a+") as file:
            kwargs["optimizer"] = str(nn.optimizer)
            kwargs["loss_fn"] = str(nn.loss_fn)
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
    for i in TEST_MODELS:
        print(i)
        parameters = {
            **TEST_MODELS[i],
            "name": i,
        }
        main(**parameters)
