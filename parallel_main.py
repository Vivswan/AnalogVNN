import argparse
import hashlib
import inspect
import itertools
import os
import subprocess
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import torchvision

from crc.analog_vnn_1 import full_parameters_list
from nn.layers.Normalize import *


def prepare_data_folder(folder_path):
    folder_path = Path(folder_path)
    runtime_path = folder_path.joinpath("runtime")
    datasets_path = folder_path.joinpath("datasets")
    models_path = folder_path.joinpath("models")
    tensorboard_path = folder_path.joinpath("tensorboard")
    logs_path = folder_path.joinpath("logs")

    if not folder_path.exists():
        os.mkdir(folder_path)
    if not runtime_path.exists():
        os.mkdir(runtime_path)
    if not datasets_path.exists():
        os.mkdir(datasets_path)
    if not models_path.exists():
        os.mkdir(models_path)
    if not tensorboard_path.exists():
        os.mkdir(tensorboard_path)
    if not logs_path.exists():
        os.mkdir(logs_path)

    torchvision.datasets.MNIST(root=str(datasets_path.absolute()), download=True)
    torchvision.datasets.FashionMNIST(root=str(datasets_path.absolute()), download=True)
    torchvision.datasets.CIFAR10(root=str(datasets_path.absolute()), download=True)
    torchvision.datasets.CIFAR100(root=str(datasets_path.absolute()), download=True)


def run_command(command):
    data_folder, command = command
    runtime = Path(data_folder).joinpath("runtime")
    filename = "std_"
    # filename += str(int(time.time())) + "_"
    filename += hashlib.sha256(str(command).encode("utf-8")).hexdigest()
    out_file = runtime.joinpath(f"{filename}.log")

    with open(out_file, "w+") as out:
        out.write(command + "\n")
        out.write(f"Running {filename} :: {command}\n\n")
        print(f"Running {filename} :: {command}")

        p = subprocess.Popen(command, shell=True, stdout=out, stderr=out)
        p.wait()
        rc = p.returncode

        out.write(f"\n\n")
        if rc == 0:
            out.write(f"Success {p.pid} :: {filename} :: {command}")
            print(f"Success {p.pid} :: {filename} :: {command}")
        else:
            out.write(f"Failed  {p.pid} :: {filename} :: {rc} :: {command}")
            print(f"Failed  {p.pid} :: {filename} :: {rc} :: {command}")

        out.write(f"\n\n{rc}")


def parameter_checkers(parameters):
    if ("norm_class" not in parameters or parameters["norm_class"] is None) and "approach" in parameters:
        if parameters["approach"] != "default":
            return False

    return True


def create_command_list(data_folder, combination_dict, test_run=False):
    combinations = list(itertools.product(*list(combination_dict.values())))
    command_list = []
    for c in combinations:
        command_dict = dict(zip(list(combination_dict.keys()), c))

        if not parameter_checkers(command_dict):
            continue

        command_str = f'python main.py --data_folder "{data_folder}"'
        if test_run:
            command_str += " --test_run true"
        for key, value in command_dict.items():
            if value is None:
                continue
            if inspect.isclass(value):
                command_str += f' --{key} "{value.__name__}"'
            elif isinstance(value, str):
                command_str += f' --{key} "{value}"'
            else:
                command_str += f' --{key} {value}'

        command_list.append(command_str)
    return command_list


def run_combination_1(data_folder, test_run=False):
    combination_dict = OrderedDict({
        "dataset": full_parameters_list["dataset"],

        "num_layer": full_parameters_list["nn_model_params"]["num_layer"],
        "activation_class": full_parameters_list["nn_model_params"]["activation_class"],
        "norm_class": full_parameters_list["nn_model_params"]["norm_class"],
        "approach": full_parameters_list["nn_model_params"]["approach"],

        "w_norm_class": full_parameters_list["weight_model_params"]["norm_class"],
    })
    return create_command_list(data_folder, combination_dict, test_run)


def run_combination_2(data_folder, test_run=False):
    activation_class = full_parameters_list["nn_model_params"]["activation_class"]
    activation_class.remove(None)
    norm_class = [Clamp, L2Norm]
    precision_class = full_parameters_list["nn_model_params"]["precision_class"]
    precision_class.remove(None)

    combination_dict = OrderedDict({
        "dataset": full_parameters_list["dataset"],

        "num_layer": full_parameters_list["nn_model_params"]["num_layer"],
        "activation_class": activation_class,
        "norm_class": norm_class,
        "precision_class": precision_class,
        "precision": full_parameters_list["nn_model_params"]["precision"],

        "w_norm_class": norm_class,
        "w_precision_class": precision_class,
        "w_precision": full_parameters_list["weight_model_params"]["precision"],
    })
    return create_command_list(data_folder, combination_dict, test_run)


def run_combination_main():
    list_combination_fn = [run_combination_1, run_combination_2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--run_combination", type=int, required=True)
    parser.add_argument("--test_run", type=bool, default=False)
    kwargs = vars(parser.parse_known_args()[0])

    print(f"test_run: {kwargs['test_run']}")
    print(f"num_process: {kwargs['num_process']}")
    print(f"data_folder: {kwargs['data_folder']}")
    print(f"run_combination: {kwargs['run_combination']}")
    prepare_data_folder(kwargs['data_folder'])

    command_list = list_combination_fn[kwargs['run_combination'] - 1](kwargs['data_folder'], kwargs["test_run"])
    TOTAL_COUNT = len(command_list)
    print(f"number of combinations: {TOTAL_COUNT}")
    print()
    print()

    # for i in command_list:
    #     print(i)
    command_list = [(kwargs['data_folder'], x) for x in command_list]
    with Pool(kwargs["num_process"]) as pool:
        pool.map(run_command, command_list)


if __name__ == '__main__':
    run_combination_main()
