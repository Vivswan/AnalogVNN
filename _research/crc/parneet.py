import argparse
import json

from _research.crc.parneet_model import run_parneet_model, RunParametersParneet
from _research.utils.path_functions import get_relative_path
from nn.layers.activations.ELU import ELU
from nn.layers.activations.Gaussian import GeLU
from nn.layers.activations.ReLU import ReLU, LeakyReLU
from nn.layers.activations.SiLU import SiLU
from nn.layers.activations.Tanh import Tanh
from nn.layers.functionals.Normalize import *
from nn.layers.functionals.ReducePrecision import ReducePrecision
from nn.layers.functionals.StochasticReducePrecision import StochasticReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.layers.noise.PoissonNoise import PoissonNoise

parneet_parameters_list = {
    "nn_model_params": {
        "activation_class": [None, ReLU, LeakyReLU, Tanh, ELU, SiLU, GeLU],
        "norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW, L1NormM, L2NormM, L1NormWM, L2NormWM],
        "precision_class": [None, ReducePrecision, StochasticReducePrecision],
        "noise_class": [None, GaussianNoise, PoissonNoise],
    },
    "weight_model_params": {
        "norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW, L1NormM, L2NormM, L1NormWM, L2NormWM],
        "precision_class": [None, ReducePrecision, StochasticReducePrecision],
        "noise_class": [None, GaussianNoise, PoissonNoise],
    },
}


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


def __check(main_obj, first, second):
    if getattr(main_obj, first) is None:
        if getattr(main_obj, second) is not None:
            raise Exception(f'{first}=None then {second} must be None')
    else:
        if getattr(main_obj, second) is None:
            raise Exception(f'{second} must not be None')


def run_parneet(kwargs):
    parameters = RunParametersParneet()

    for key, value in kwargs.items():
        if hasattr(parameters, key):
            setattr(parameters, key, value)

    __select_class(parameters, 'activation_class', parneet_parameters_list["nn_model_params"]["activation_class"])
    __select_class(parameters, 'norm_class', parneet_parameters_list["nn_model_params"]["norm_class"])
    __select_class(parameters, 'precision_class', parneet_parameters_list["nn_model_params"]["precision_class"])
    __select_class(parameters, 'noise_class', parneet_parameters_list["nn_model_params"]["noise_class"])
    __select_class(parameters, 'w_norm_class', parneet_parameters_list["weight_model_params"]["norm_class"])
    __select_class(parameters, 'w_precision_class', parneet_parameters_list["weight_model_params"]["precision_class"])
    __select_class(parameters, 'w_noise_class', parneet_parameters_list["weight_model_params"]["noise_class"])

    __check(parameters, "precision_class", "precision")
    __check(parameters, "noise_class", "leakage")
    __check(parameters, "w_precision_class", "w_precision")
    __check(parameters, "w_noise_class", "w_leakage")

    # print(parameters)
    run_parneet_model(parameters)


def parser_run_parneet(main_file=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--data_folder", type=str, required=True)

    parser.add_argument("--activation_class", type=str, default=None)
    parser.add_argument("--norm_class", type=str, default=None)
    parser.add_argument("--precision_class", type=str, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--noise_class", type=str, default=None)
    parser.add_argument("--leakage", type=float, default=None)

    parser.add_argument("--w_norm_class", type=str, default=None)
    parser.add_argument("--w_precision_class", type=str, default=None)
    parser.add_argument("--w_precision", type=int, default=None)
    parser.add_argument("--w_noise_class", type=str, default=None)
    parser.add_argument("--w_leakage", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=RunParametersParneet.epochs)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--color", type=str, default=str(RunParametersParneet.color))
    parser.add_argument("--batch_size", type=int, default=RunParametersParneet.batch_size)

    parser.add_argument("--test_logs", action='store_true')
    parser.set_defaults(test_logs=False)
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
    if kwargs["color"].lower() == "true":
        kwargs["color"] = True
    elif kwargs["color"].lower() == "false":
        kwargs["color"] = False
    else:
        raise ValueError("Invalid value for color")

    run_parneet(kwargs)


if __name__ == '__main__':
    parser_run_parneet()
