import os
import shutil
import time
from dataclasses import dataclass

from utils.path_functions import path_join


@dataclass
class DataPaths:
    name: str
    model_data: str
    tensorboard: str
    dataset: str
    logs: str
    timestamp: str


def data_dirs(path, name=None, timestamp=None, tensorboard=True):
    if timestamp is None:
        timestamp = str(int(time.time()))

    name = timestamp + ("" if name is None else ("_" + name))

    dataset_path = path_join(path, "datasets")
    logs_path = path_join(path, "logs")

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    if not os.path.exists(path_join(path, "models")):
        os.mkdir(path_join(path, "models"))
    if not os.path.exists(path_join(path, "tensorboard")):
        os.mkdir(path_join(path, "tensorboard"))

    models_path = path_join(path, f"models/{name}")
    tensorboard_path = path_join(path, f"tensorboard/{name}")

    if tensorboard:
        os.mkdir(tensorboard_path)

    if models_path:
        os.mkdir(models_path)

    return DataPaths(
        name=name,
        model_data=models_path,
        tensorboard=tensorboard_path,
        dataset=dataset_path,
        logs=logs_path,
        timestamp=timestamp,
    )


def erase_data_dirs(path):
    if os.path.exists(path):
        if os.path.exists(path_join(path, "models")):
            shutil.rmtree(path_join(path, "models"))
        if os.path.exists(path_join(path, "tensorboard")):
            shutil.rmtree(path_join(path, "tensorboard"))
