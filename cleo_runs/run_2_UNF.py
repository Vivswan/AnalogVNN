import os
import time

import torchvision

from cleo_runs.run_2 import *
from nn.activations.ELU import ELU
from nn.activations.Gaussian import GeLU
from nn.activations.Identity import Identity
from nn.activations.ReLU import ReLU, LeakyReLU
from nn.activations.SiLU import SiLU
from nn.activations.Tanh import Tanh
from nn.layers.Normalize import Clamp, L2Norm

if __name__ == '__main__':
    DATA_FOLDER = f"C:/_data/tensorboard_cleo_{os.path.basename(__file__)}/"
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    timestamp = str(int(time.time()))

    approach = "use_norm_forward"
    ticker = "".join([i[0].upper() for i in approach.split("_")])
    cleo_default_parameters["data_folder"] = DATA_FOLDER
    cleo_default_parameters["approach"] = approach
    parameters = combination_parameters_sequence(
        {ticker: cleo_default_parameters},
        ("model_class", [Linear2, Linear3, Linear4]),
        ("activation_class", [ReLU, Identity, LeakyReLU, Tanh, ELU, SiLU, GeLU]),
        ("norm_class_y", [L2Norm, Clamp]),
        ("norm_class_w", [Clamp, L2Norm]),
        ("precision_y", [2 ** 2, 2 ** 4, 2 ** 4, None]),
        ("precision_w", [2 ** 2, 2 ** 4, 2 ** 4, None]),
        ("dataset", [torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST]),
    )
    print(len(parameters.keys()))
    run_function_with_parameters(main, parameters, continue_from="")
