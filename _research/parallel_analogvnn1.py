import argparse
import copy
import hashlib
import inspect
import itertools
import math
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torchvision
from natsort import natsorted, ns

from _research.crc.analog_vnn_1 import analogvnn1_parameters_list
from nn.layers.functionals.Normalize import *


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
    hash_id = hashlib.sha256(str(command).encode("utf-8")).hexdigest()
    timestamp = f"{int(time.time() * 1000)}"

    if "--timestamp" not in command:
        command += f" --timestamp {timestamp}"
    else:
        timestamp = command.split("--timestamp")[-1]
        timestamp = timestamp.strip().split(" ")[0]

    if "--name" not in command:
        command += f" --name {hash_id}"
    else:
        hash_id = command.split("--name")[-1]
        hash_id = hash_id.strip().split(" ")[0]

    filename = f"{timestamp}_{hash_id}"
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


def create_command_list(data_folder, combination_dict, extra_arg="", select=""):
    if len(select) > 0:
        for parameter in select.split(","):
            parameter = parameter.strip()
            key, value = parameter.split(":")
            values = combination_dict[key]
            for v in values:
                if value in str(v):
                    combination_dict[key] = [v]
                    break

    combinations = list(itertools.product(*list(combination_dict.values())))
    command_list = []
    for c in combinations:
        command_dict = dict(zip(list(combination_dict.keys()), c))

        if not parameter_checkers(command_dict):
            continue

        command_str = f'python main_analogvnn1.py --data_folder "{data_folder}"'
        for key, value in command_dict.items():
            if value is None:
                continue
            if inspect.isclass(value):
                command_str += f' --{key} "{value.__name__}"'
            elif isinstance(value, str):
                command_str += f' --{key} "{value}"'
            else:
                command_str += f' --{key} {value}'

        command_str += f" {extra_arg}"
        command_list.append(command_str)
    command_list = natsorted(command_list, alg=ns.IGNORECASE)
    return command_list


def run_combination_1(data_folder, extra_arg="", select=""):
    combination_dict = OrderedDict({
        "dataset": analogvnn1_parameters_list["dataset"],

        "num_conv_layer": analogvnn1_parameters_list["nn_model_params"]["num_conv_layer"],
        "num_linear_layer": analogvnn1_parameters_list["nn_model_params"]["num_linear_layer"],
        "activation_class": analogvnn1_parameters_list["nn_model_params"]["activation_class"],
        "norm_class": analogvnn1_parameters_list["nn_model_params"]["norm_class"],
        "approach": analogvnn1_parameters_list["nn_model_params"]["approach"],

        "w_norm_class": analogvnn1_parameters_list["weight_model_params"]["norm_class"],
    })
    return create_command_list(data_folder, combination_dict, extra_arg=extra_arg, select=select)


def run_combination_2(data_folder, extra_arg="", select=""):
    activation_class = copy.deepcopy(analogvnn1_parameters_list["nn_model_params"]["activation_class"])
    activation_class.remove(None)
    precision_class = copy.deepcopy(analogvnn1_parameters_list["nn_model_params"]["precision_class"])
    precision_class.remove(None)
    norm_class = [Clamp, L2Norm]

    combination_dict = OrderedDict({
        "dataset": analogvnn1_parameters_list["dataset"],

        "num_conv_layer": analogvnn1_parameters_list["nn_model_params"]["num_conv_layer"],
        "num_linear_layer": analogvnn1_parameters_list["nn_model_params"]["num_linear_layer"],
        "activation_class": activation_class,
        "norm_class": norm_class,
        "precision_class": precision_class,
        "precision": analogvnn1_parameters_list["nn_model_params"]["precision"],

        "w_norm_class": norm_class,
        "w_precision_class": precision_class,
        "w_precision": analogvnn1_parameters_list["weight_model_params"]["precision"],
    })
    return create_command_list(data_folder, combination_dict, extra_arg=extra_arg, select=select)


def run_combination_3(data_folder, extra_arg="", select=""):
    activation_class = copy.deepcopy(analogvnn1_parameters_list["nn_model_params"]["activation_class"])
    activation_class.remove(None)
    precision_class = copy.deepcopy(analogvnn1_parameters_list["nn_model_params"]["precision_class"])
    precision_class.remove(None)
    norm_class = [Clamp]
    noise_class = copy.deepcopy(analogvnn1_parameters_list["nn_model_params"]["noise_class"])
    noise_class.remove(None)

    combination_dict = OrderedDict({
        "dataset": analogvnn1_parameters_list["dataset"],

        "num_conv_layer": analogvnn1_parameters_list["nn_model_params"]["num_conv_layer"],
        "num_linear_layer": analogvnn1_parameters_list["nn_model_params"]["num_linear_layer"],
        "activation_class": activation_class,
        "norm_class": norm_class,
        "precision_class": precision_class,
        "precision": analogvnn1_parameters_list["nn_model_params"]["precision"],
        "noise_class": noise_class,
        "leakage": analogvnn1_parameters_list["nn_model_params"]["leakage"],

        "w_norm_class": norm_class,
        "w_precision_class": precision_class,
        "w_precision": analogvnn1_parameters_list["weight_model_params"]["precision"],
        "w_noise_class": noise_class,
        "w_leakage": analogvnn1_parameters_list["weight_model_params"]["leakage"],
    })
    return create_command_list(data_folder, combination_dict, extra_arg=extra_arg, select=select)


RUN_UNDER_12_LIST = {
    "11lm": (run_combination_1, "num_linear_layer:1,num_conv_layer:0,approach:default,dataset:MNIST"),
    "11cm": (run_combination_1, "num_linear_layer:1,num_conv_layer:3,approach:default,dataset:MNIST"),
    "11lf": (run_combination_1, "num_linear_layer:1,num_conv_layer:0,approach:default,dataset:FashionMNIST"),
    "11cf": (run_combination_1, "num_linear_layer:1,num_conv_layer:3,approach:default,dataset:FashionMNIST"),
    "11lc": (run_combination_1, "num_linear_layer:1,num_conv_layer:0,approach:default,dataset:CIFAR10"),
    "11cc": (run_combination_1, "num_linear_layer:1,num_conv_layer:3,approach:default,dataset:CIFAR10"),
    "12lm": (run_combination_1, "num_linear_layer:2,num_conv_layer:0,approach:default,dataset:MNIST"),
    "12cm": (run_combination_1, "num_linear_layer:2,num_conv_layer:3,approach:default,dataset:MNIST"),
    "12lf": (run_combination_1, "num_linear_layer:2,num_conv_layer:0,approach:default,dataset:FashionMNIST"),
    "12cf": (run_combination_1, "num_linear_layer:2,num_conv_layer:3,approach:default,dataset:FashionMNIST"),
    "12lc": (run_combination_1, "num_linear_layer:2,num_conv_layer:0,approach:default,dataset:CIFAR10"),
    "12cc": (run_combination_1, "num_linear_layer:2,num_conv_layer:3,approach:default,dataset:CIFAR10"),
    "13lm": (run_combination_1, "num_linear_layer:3,num_conv_layer:0,approach:default,dataset:MNIST"),
    "13cm": (run_combination_1, "num_linear_layer:3,num_conv_layer:3,approach:default,dataset:MNIST"),
    "13lf": (run_combination_1, "num_linear_layer:3,num_conv_layer:0,approach:default,dataset:FashionMNIST"),
    "13cf": (run_combination_1, "num_linear_layer:3,num_conv_layer:3,approach:default,dataset:FashionMNIST"),
    "13lc": (run_combination_1, "num_linear_layer:3,num_conv_layer:0,approach:default,dataset:CIFAR10"),
    "13cc": (run_combination_1, "num_linear_layer:3,num_conv_layer:3,approach:default,dataset:CIFAR10"),

    # "2lml": (run_combination_2, "num_conv_layer:0,dataset:MNIST,norm_class:L2Norm"),
    # "2cml": (run_combination_2, "num_conv_layer:3,dataset:MNIST,norm_class:L2Norm"),
    # "2lfl": (run_combination_2, "num_conv_layer:0,dataset:FashionMNIST,norm_class:L2Norm"),
    # "2cfl": (run_combination_2, "num_conv_layer:3,dataset:FashionMNIST,norm_class:L2Norm"),
    # "2lcl": (run_combination_2, "num_conv_layer:0,dataset:CIFAR10,norm_class:L2Norm"),
    # "2ccl": (run_combination_2, "num_conv_layer:3,dataset:CIFAR10,norm_class:L2Norm"),
    # "2lmc": (run_combination_2, "num_conv_layer:0,dataset:MNIST,norm_class:Clamp"),
    # "2cmc": (run_combination_2, "num_conv_layer:3,dataset:MNIST,norm_class:Clamp"),
    # "2lfc": (run_combination_2, "num_conv_layer:0,dataset:FashionMNIST,norm_class:Clamp"),
    # "2cfc": (run_combination_2, "num_conv_layer:3,dataset:FashionMNIST,norm_class:Clamp"),
    # "2lcc": (run_combination_2, "num_conv_layer:0,dataset:CIFAR10,norm_class:Clamp"),
    # "2ccc": (run_combination_2, "num_conv_layer:3,dataset:CIFAR10,norm_class:Clamp"),
    #
    # "3glmr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:MNIST,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    # "3gcmr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:MNIST,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    # "3glfr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:FashionMNIST,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    # "3gcfr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:FashionMNIST,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    # "3glcr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:CIFAR10,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    # "3gccr": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:CIFAR10,precision_class:ReducePrecision,w_precision_class:ReducePrecision"
    # ),
    #
    # "3glms": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:MNIST,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),
    # "3gcms": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:MNIST,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),
    # "3glfs": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:FashionMNIST,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),
    # "3gcfs": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:FashionMNIST,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),
    # "3glcs": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:0,dataset:CIFAR10,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),
    # "3gccs": (
    #     run_combination_3,
    #     "noise_class:GaussianNoise,w_noise_class:GaussianNoise,"
    #     "num_conv_layer:3,dataset:CIFAR10,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"
    # ),

    # "3lmpr": (run_combination_3,
    #           "num_conv_layer:0,dataset:MNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),
    # "3cmpr": (run_combination_3,
    #           "num_conv_layer:3,dataset:MNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),
    # "3lfpr": (run_combination_3,
    #           "num_conv_layer:0,dataset:FashionMNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),
    # "3cfpr": (run_combination_3,
    #           "num_conv_layer:3,dataset:FashionMNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),
    # "3lcpr": (run_combination_3,
    #           "num_conv_layer:0,dataset:CIFAR10,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),
    # "3ccpr": (run_combination_3,
    #           "num_conv_layer:3,dataset:CIFAR10,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:ReducePrecision,w_precision_class:ReducePrecision"),

    # "3lmps": (run_combination_3,
    #           "num_conv_layer:0,dataset:MNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
    # "3cmps": (run_combination_3,
    #           "num_conv_layer:3,dataset:MNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
    # "3lfps": (run_combination_3,
    #           "num_conv_layer:0,dataset:FashionMNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
    # "3cfps": (run_combination_3,
    #           "num_conv_layer:3,dataset:FashionMNIST,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
    # "3lcps": (run_combination_3,
    #           "num_conv_layer:0,dataset:CIFAR10,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
    # "3ccps": (run_combination_3,
    #           "num_conv_layer:3,dataset:CIFAR10,noise_class:PoissonNoise,w_noise_class:GaussianNoise,precision_class:StochasticReducePrecision,w_precision_class:StochasticReducePrecision"),
}


def run_combination_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_required", type=int, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--select", type=str, default="")
    parser.add_argument("--run_combination", type=str, required=True)
    parser.add_argument("--run_index", type=int, default=-1)
    parser.add_argument("--single_run", action='store_true')
    parser.set_defaults(single_run=False)
    all_arguments = parser.parse_known_args()
    print(all_arguments)

    kwargs = vars(all_arguments[0])
    extra_arg = ""
    for i in all_arguments[1]:
        if " " in i:
            extra_arg += f' "{i}"'
        else:
            extra_arg += f' {i}'

    if torch.cuda.is_available():
        num_process = max(math.floor(torch.cuda.mem_get_info()[1] / (kwargs['memory_required'] * 1024 ** 2)), 1)
    else:
        num_process = 1

    print(f"memory_required: {kwargs['memory_required']}")
    print(f"num_process: {num_process}")
    print(f"data_folder: {kwargs['data_folder']}")
    print(f"run_combination: {kwargs['run_combination']}")
    print(f"run_index: {kwargs['run_index']}")
    prepare_data_folder(kwargs['data_folder'])

    select = RUN_UNDER_12_LIST[kwargs['run_combination']][1]
    if len(kwargs['select']) > 0:
        select += "," + kwargs['select']

    command_list = RUN_UNDER_12_LIST[kwargs['run_combination']][0](kwargs['data_folder'], extra_arg, select)
    print(f"number of combinations: {len(command_list)}")
    print()
    print()

    command_list = [(kwargs['data_folder'], x) for x in command_list]
    if kwargs['run_index'] >= 0:
        command_list = [command_list[kwargs['run_index'] - 1]]

    if kwargs["single_run"]:
        command_list = command_list[:kwargs["num_process"]]

    with ThreadPool(num_process) as pool:
        pool.map(run_command, command_list)


def create_slurm_scripts():
    shutil.rmtree("_crc_slurm")
    os.mkdir("_crc_slurm")
    with open("run_array_@@@.slurm", "r") as main_run_file:
        text = main_run_file.read()

    for i in RUN_UNDER_12_LIST:
        with open(f"_crc_slurm/run_analogvnn1_{i}.slurm", "w") as file:
            file.write(
                text
                .replace("@@@cpu@@@", "3")
                .replace("@@@partition@@@", "scavenger")
                .replace("@@@time@@@", "1-00:00:00")
                .replace("@@@VideoMemoryRequired@@@", "11000")
                .replace("@@@RunScript@@@", Path(__file__).name.split(".")[0])
                .replace("@@@run_combination@@@", i)
                .replace("@@@array@@@", f"1-{len(RUN_UNDER_12_LIST[i][0]('', '', RUN_UNDER_12_LIST[i][1]))}")
                .replace("@@@extra@@@", "")
            )
            # file.write(
            #     text
            #         .replace("@@@cpu@@@", "16")
            #         .replace("@@@partition@@@", "a100")
            #         .replace("@@@time@@@", "2-00:00:00")
            #         .replace("@@@VideoMemoryRequired@@@", "1800")
            #         .replace("@@@RunScript@@@", Path(__file__).name.split(".")[0])
            #         .replace("@@@run_combination@@@", i)
            # )

    with open(f"_crc_slurm/_run.sh", "w") as file:
        for i in RUN_UNDER_12_LIST:
            file.write(f"sbatch run_analogvnn1_{i}.slurm\n")


if __name__ == '__main__':
    create_slurm_scripts()
    for name, value in RUN_UNDER_12_LIST.items():
        size = len(value[0]('', '', value[1]))
        print(f"{name}: {size}, {size * 0.01080530071}")
    print()
    # run_combination_main()
