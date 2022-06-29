import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Type

import torchvision
from torch import optim
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset

from crc.analog_vnn_1_model import run_main_model
from nn.activations.Activation import Activation
from nn.activations.ELU import ELU
from nn.activations.Gaussian import GeLU
from nn.activations.ReLU import ReLU, LeakyReLU
from nn.activations.SiLU import SiLU
from nn.activations.Tanh import Tanh
from nn.layers.functionals.Normalize import *
from nn.layers.functionals.ReducePrecision import ReducePrecision
from nn.layers.functionals.StochasticReducePrecision import StochasticReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.layers.noise.PoissonNoise import PoissonNoise
from utils.path_functions import get_relative_path

full_parameters_list = {
    "nn_model_params": {
        "num_conv_layer": [0, 3],
        "num_linear_layer": [1, 2, 3],
        "activation_class": [None, ReLU, LeakyReLU, Tanh, ELU, SiLU, GeLU],

        "norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW],
        "approach": ["default", "no_norm_grad"],

        "precision_class": [None, ReducePrecision, StochasticReducePrecision],
        "precision": [2 ** 2, 2 ** 4, 2 ** 6],

        "noise_class": [None, GaussianNoise, PoissonNoise],
        "leakage": [0.25, 0.5, 0.75],
    },
    "weight_model_params": {
        "norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW],

        "precision_class": [None, ReducePrecision, StochasticReducePrecision],
        "precision": [2 ** 2, 2 ** 4, 2 ** 6],

        "noise_class": [None, GaussianNoise, PoissonNoise],
        "leakage": [0.25, 0.5, 0.75],
    },
    "optimiser_class": optim.Adam,
    "optimiser_parameters": {},
    "dataset": [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.CIFAR10,
        # torchvision.datasets.CIFAR100,
    ],
    "batch_size": 1280,
    "epochs": 10,
    "test_run": False,
}

LINEAR_LAYER_SIZES = {
    1: [],
    2: [64],
    3: [128, 64],
    4: [256, 128, 64]
}
CONV_LAYER_SIZES = {
    0: [],
    3: [(1, 32, (3, 3)), (32, 64, (3, 3)), (64, 64, (3, 3))],
}


@dataclass
class RunParameters:
    name: Union[None, str] = None
    data_folder: Union[None, str] = None

    num_conv_layer: Union[None, int] = 0
    num_linear_layer: Union[None, int] = 1
    activation_class: Union[None, Type[Activation]] = None
    norm_class: Union[None, Type[Normalize]] = None
    approach: Union[None, str] = "default"
    precision_class: Type[BaseLayer] = None
    precision: Union[None, int] = None
    noise_class: Type[BaseLayer] = None
    leakage: Union[None, float] = None

    w_norm_class: Union[None, Type[Normalize]] = None
    w_precision_class: Type[BaseLayer] = None
    w_precision: Union[None, int] = None
    w_noise_class: Type[BaseLayer] = None
    w_leakage: Union[None, float] = None

    optimiser_class: Type[Optimizer] = optim.Adam
    optimiser_parameters: dict = None
    dataset: Type[VisionDataset] = torchvision.datasets.MNIST
    batch_size: int = 128
    epochs: int = 10

    device: Union[None, torch.device] = None
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    def __init__(self):
        self.optimiser_parameters = {}

    @property
    def nn_model_params(self):
        return {
            "num_conv_layer": self.num_conv_layer,
            "num_linear_layer": self.num_linear_layer,
            "activation_class": self.activation_class,
            "norm_class": self.norm_class,
            "approach": self.approach,
            "precision_class": self.precision_class,
            "precision": self.precision,
            "noise_class": self.noise_class,
            "leakage": self.leakage,
        }

    @property
    def weight_model_params(self):
        return {
            "norm_class": self.w_norm_class,
            "precision_class": self.w_precision_class,
            "precision": self.w_precision,
            "noise_class": self.w_noise_class,
            "leakage": self.w_leakage,
        }

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"RunParameters({json.dumps(self.json)})"


def __select_class(main_object, name, class_list):
    value = getattr(main_object, name)
    if value is None and None in class_list:
        setattr(main_object, name, None)
        return None
    if value is None and None not in class_list:
        raise Exception(f"{name} must be in {class_list}")

    for cl in class_list:
        if cl is None:
            continue
        if value == cl:
            setattr(main_object, name, cl)
            return cl
        if isinstance(value, str):
            if value == cl.__name__:
                setattr(main_object, name, cl)
                return cl

    raise Exception(f"{name} must be in {class_list}")


def __check(main_obj, first, second, check_list):
    if getattr(main_obj, first) is None:
        if getattr(main_obj, second) is not None:
            raise Exception(f'{first}=None then {second} must be None')
    else:
        if getattr(main_obj, second) not in check_list:
            raise Exception(f'{second} must be in {check_list}')


def run_main(kwargs):
    parameters = RunParameters()

    for key, value in kwargs.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)

    __select_class(parameters, 'dataset', full_parameters_list["dataset"])
    __select_class(parameters, 'activation_class', full_parameters_list["nn_model_params"]["activation_class"])
    __select_class(parameters, 'norm_class', full_parameters_list["nn_model_params"]["norm_class"])
    __select_class(parameters, 'precision_class', full_parameters_list["nn_model_params"]["precision_class"])
    __select_class(parameters, 'noise_class', full_parameters_list["nn_model_params"]["noise_class"])
    __select_class(parameters, 'w_norm_class', full_parameters_list["weight_model_params"]["norm_class"])
    __select_class(parameters, 'w_precision_class', full_parameters_list["weight_model_params"]["precision_class"])
    __select_class(parameters, 'w_noise_class', full_parameters_list["weight_model_params"]["noise_class"])

    if parameters.num_conv_layer not in full_parameters_list["nn_model_params"]["num_conv_layer"]:
        raise Exception(f'num_conv_layer must be in {full_parameters_list["nn_model_params"]["num_conv_layer"]}')
    if parameters.num_linear_layer not in full_parameters_list["nn_model_params"]["num_linear_layer"]:
        raise Exception(f'num_linear_layer must be in {full_parameters_list["nn_model_params"]["num_linear_layer"]}')
    if parameters.approach not in full_parameters_list["nn_model_params"]["approach"]:
        raise Exception(f'approach must be in {full_parameters_list["nn_model_params"]["approach"]}')

    if parameters.norm_class is None and parameters.approach != "default":
        raise Exception('norm_class=None then approach must be "default"')

    __check(parameters, "precision_class", "precision", full_parameters_list["nn_model_params"]["precision"])
    __check(parameters, "noise_class", "leakage", full_parameters_list["nn_model_params"]["leakage"])
    __check(parameters, "w_precision_class", "w_precision", full_parameters_list["weight_model_params"]["precision"])
    __check(parameters, "w_noise_class", "w_leakage", full_parameters_list["weight_model_params"]["leakage"])

    # print(parameters)
    run_main_model(parameters)


def parser_run_main(main_file=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--num_conv_layer", type=int, required=True)
    parser.add_argument("--num_linear_layer", type=int, required=True)
    parser.add_argument("--activation_class", type=str, default=None)

    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--approach", type=str, default="default")
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--w_norm_class", type=str, default=None)
    parser.add_argument("--w_precision_class", type=str, default=None)
    parser.add_argument("--w_precision", type=int, default=None)
    parser.add_argument("--w_noise_class", type=str, default=None)
    parser.add_argument("--w_leakage", type=float, default=None)

    parser.add_argument("--dataset", type=str, default="MNIST")
    # parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--test_run", action='store_true')
    parser.set_defaults(test_run=False)
    parser.add_argument("--tensorboard", action='store_true')
    parser.set_defaults(tensorboard=False)
    parser.add_argument("--save_data", action='store_true')
    parser.set_defaults(save_data=False)
    kwargs = vars(parser.parse_known_args()[0])
    print(json.dumps(kwargs))
    print()

    if main_file is None:
        main_file = __file__

    kwargs["data_folder"] = get_relative_path(main_file, kwargs["data_folder"])

    run_main(kwargs)


if __name__ == '__main__':
    parser_run_main()
