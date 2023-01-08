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

import torch
import torchvision
from natsort import natsorted, ns

from analogvnn.nn.activation.ELU import ELU
from analogvnn.nn.activation.Gaussian import GeLU
from analogvnn.nn.activation.ReLU import LeakyReLU, ReLU
from analogvnn.nn.activation.SiLU import SiLU
from analogvnn.nn.activation.Tanh import Tanh
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import *
from analogvnn.nn.normalize.LPNorm import L1Norm, L2Norm, L1NormW, L2NormW, L1NormM, L2NormM, L1NormWM, L2NormWM
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.nn.precision.StochasticReducePrecision import StochasticReducePrecision

combination_dict = OrderedDict({
    "color": [False, True],
    "activation_class": [ReLU, LeakyReLU, SiLU, ELU, GeLU, Tanh],
    "norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW, L1NormM, L2NormM, L1NormWM, L2NormWM],
    "w_norm_class": [None, Clamp, L1Norm, L2Norm, L1NormW, L2NormW, L1NormM, L2NormM, L1NormWM, L2NormWM],
    "batch_size": [128, 256, 384, 512],
    "precision_class": [None, ReducePrecision, StochasticReducePrecision],
    "noise_class": [None, GaussianNoise],

    "precision": [None, 4, 8, 16, 32, 64],
    "w_precision": [None, 4, 8, 16, 32, 64],
    "leakage": [None, 0.2, 0.4, 0.6, 0.8],
    "w_leakage": [None, 0.2, 0.4, 0.6, 0.8],
})

RUN_PARNEET_LIST = {
    # "all": "",
    # b
    # "nb": "norm_class:None,w_norm_class:None,precision_class:None",
    # "cb": "norm_class:Clamp,w_norm_class:Clamp,precision_class:None",

    # n
    "ng": "precision_class:None,batch_size:512,color:False",
    "nc": "precision_class:None,batch_size:512,color:True",

    # p
    # "lp": "activation_class:LeakyReLU,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "gp": "activation_class:GeLU,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "rp": "activation_class:ReLU,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "sp": "activation_class:SiLU,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "ep": "activation_class:ELU,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "tp": "activation_class:Tanh,precision_class:~None,noise_class:None,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",

    # g
    # "lrg": "activation_class:LeakyReLU,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "lsg": "activation_class:LeakyReLU,precision_class:StochasticReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "grg": "activation_class:GeLU,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    "gsg": "activation_class:GeLU,precision_class:StochasticReducePrecision,"
           "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "rrg": "activation_class:ReLU,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "rsg": "activation_class:ReLU,precision_class:StochasticReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "srg": "activation_class:SiLU,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "ssg": "activation_class:SiLU,precision_class:StochasticReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "erg": "activation_class:ELU,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "esg": "activation_class:ELU,precision_class:StochasticReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "trg": "activation_class:Tanh,precision_class:ReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
    # "tsg": "activation_class:Tanh,precision_class:StochasticReducePrecision,"
    #        "noise_class:GaussianNoise,norm_class:Clamp,w_norm_class:Clamp,batch_size:512",
}


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

    torchvision.datasets.CIFAR10(root=str(datasets_path.absolute()), download=True)


def run_command(command):
    data_folder, command = command
    runtime = Path(data_folder).joinpath("runtime")
    hash_id = hashlib.sha256(str(command).encode("utf-8")).hexdigest()
    timestamp = f"{int(time.time() * 1000)}"

    command += f" --data_folder {data_folder}"
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


def create_command_list(extra_arg="", select=""):
    cd_copy = copy.deepcopy(combination_dict)
    if len(select) > 0:
        for parameter in select.split(","):
            parameter = parameter.strip()
            key, value = parameter.split(":")
            not_in = value[0] == "~"
            if not_in:
                value = value[1:]

            values = cd_copy[key]
            cd_copy[key] = []
            for v in values:
                if inspect.isclass(v):
                    v = v.__name__
                if value == str(v) and not not_in:
                    cd_copy[key].append(v)
                if value != str(v) and not_in:
                    cd_copy[key].append(v)

    combinations = list(itertools.product(*list(cd_copy.values())))

    command_list = []
    for c in combinations:
        command_dict = dict(zip(list(cd_copy.keys()), c))
        # command_dict["w_norm_class"] = command_dict["norm_class"]
        command_dict["w_precision_class"] = command_dict["precision_class"]
        command_dict["w_noise_class"] = command_dict["noise_class"]

        if command_dict["norm_class"] is None and \
                (command_dict["precision_class"] is not None or command_dict["noise_class"] is not None):
            continue

        if command_dict["precision_class"] is None and command_dict["precision"] is not None:
            continue
        if command_dict["noise_class"] is None and command_dict["leakage"] is not None:
            continue
        if command_dict["w_precision_class"] is None and command_dict["w_precision"] is not None:
            continue
        if command_dict["w_noise_class"] is None and command_dict["w_leakage"] is not None:
            continue
        if command_dict["precision_class"] is not None and command_dict["precision"] is None:
            continue
        if command_dict["noise_class"] is not None and command_dict["leakage"] is None:
            continue
        if command_dict["w_precision_class"] is not None and command_dict["w_precision"] is None:
            continue
        if command_dict["w_noise_class"] is not None and command_dict["w_leakage"] is None:
            continue

        if command_dict["noise_class"] is not None and command_dict["precision"] is None:
            continue
        if command_dict["w_noise_class"] is not None and command_dict["w_precision"] is None:
            continue

        if not parameter_checkers(command_dict):
            continue

        command_str = f'python main_parneet.py'
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
        command_list.append(command_str.strip())
        # print(command_str)
    command_list = natsorted(command_list, alg=ns.IGNORECASE)
    # with open(f"_data/{list(RUN_PARNEET_LIST.keys())[list(RUN_PARNEET_LIST.values()).index(select)]}.txt", "w") as file:
    #     file.write("\n".join(command_list))
    return command_list


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

    select = RUN_PARNEET_LIST[kwargs['run_combination']]
    if len(kwargs['select']) > 0:
        select += "," + kwargs['select']

    command_list = create_command_list(extra_arg, select)
    print(f"number of combinations: {len(command_list)}")
    print()
    print()

    command_list = [(kwargs['data_folder'], x) for x in command_list]
    if kwargs['run_index'] >= 0:
        command_list = [command_list[kwargs['run_index'] - 1]]

    if kwargs["single_run"]:
        command_list = command_list[:num_process]

    with ThreadPool(num_process) as pool:
        pool.map(run_command, command_list)


def create_slurm_scripts():
    shutil.rmtree("_crc_slurm")
    os.mkdir("_crc_slurm")
    with open("run_array_@@@.slurm", "r") as main_run_file:
        text = main_run_file.read()

    with open(f"_crc_slurm/_run.sh", "w") as run_file:
        for i in RUN_PARNEET_LIST:
            with open(f"_crc_slurm/run_parneet_{i}.slurm", "w") as slurm_file:
                slurm_file.write(
                    text
                    .replace("@@@cpu@@@", "2")
                    .replace("@@@partition@@@", "scavenger")
                    .replace("@@@time@@@", "1-00:00:00")
                    .replace("@@@VideoMemoryRequired@@@", "11000")
                    .replace("@@@RunScript@@@", Path(__file__).name.split(".")[0])
                    .replace("@@@run_combination@@@", i)
                    .replace("@@@array@@@", f"1-{len(create_command_list('', RUN_PARNEET_LIST[i]))}")
                    .replace("@@@extra@@@", "")
                )

            run_file.write(f"sbatch run_parneet_{i}.slurm\n")


if __name__ == '__main__':
    create_slurm_scripts()
    total = 0
    for name, value in RUN_PARNEET_LIST.items():
        size = len(create_command_list('', value))
        total += size
        print(f"{name}: {size}, {size * 0.2197231834}")
    print(f"total: {total}")

    # run_combination_main()
