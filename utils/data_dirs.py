import os
import time

from utils.path_functions import path_join


def data_dirs(path, name=None):
    timestamp = str(int(time.time()))
    name = timestamp + ("" if name is None else ("_" + name))

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_join(path, "models")):
        os.mkdir(path_join(path, "models"))
    if not os.path.exists(path_join(path, "tensorboard")):
        os.mkdir(path_join(path, "tensorboard"))
    if not os.path.exists(path_join(path, "datasets")):
        os.mkdir(path_join(path, "datasets"))

    models_path = path_join(path, f"models/{name}")
    tensorboard_path = path_join(path, f"tensorboard/{name}")
    dataset_path = path_join(path, "datasets")

    os.mkdir(models_path)
    os.mkdir(tensorboard_path)
    return name, models_path, tensorboard_path, dataset_path
