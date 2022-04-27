import copy
import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Union

import matplotlib
import numpy as np
import seaborn as seaborn
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, close
from torch import nn
from torch.utils.data import DataLoader

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.layers.Normalize import Clamp, Clamp01
from nn.layers.ReducePrecision import ReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.utils.is_cpu_cuda import is_cpu_cuda


def collect_parameters_to_json(path, destination=None):
    from tensorboard.plugins.hparams.metadata import SESSION_START_INFO_TAG
    from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
    from tensorflow.python.summary.summary_iterator import summary_iterator
    tensorboard_dir = Path(path).joinpath("tensorboard")
    all_files: List[Path] = []
    for root, dirs, files in os.walk(tensorboard_dir):
        for file in files:
            all_files.append(Path(root).joinpath(file))
    parameter_data = {}
    # c = False
    for i, file in enumerate(all_files):
        name = file.parent
        if "_" not in str(name.name):
            name = name.parent
            # c = True
        name = name.name
        if i % 10 == 0:
            print(f"[{i}/{len(all_files)}] Processing {name}...")
        if name not in parameter_data:
            parameter_data[name] = {
                "test_accuracy": {},
                "train_accuracy": {},
                "test_loss": {},
                "train_loss": {},
                "parameters": {},
                "raw": [],
            }

        this_data = parameter_data[name]
        for event in summary_iterator(str(file)):
            for value in event.summary.value:
                if value.tag == 'Accuracy/test':
                    this_data["test_accuracy"][int(event.step)] = value.simple_value
                    continue
                if value.tag == 'Loss/test':
                    this_data["train_accuracy"][int(event.step)] = value.simple_value
                    continue
                if value.tag == 'Accuracy/train':
                    this_data["test_loss"][int(event.step)] = value.simple_value
                    continue
                if value.tag == 'Loss/train':
                    this_data["train_loss"][int(event.step)] = value.simple_value
                    continue
                if value.tag == SESSION_START_INFO_TAG:
                    ssi = HParamsPluginData()
                    ssi.ParseFromString(value.metadata.plugin_data.content)
                    hparams = dict(ssi.session_start_info.hparams)
                    for k in hparams:
                        hparams[k] = hparams[k].ListFields()[0][1]
                    this_data["parameters"] = hparams
            # this_data["raw"].append(event)

        # if c:
        # break
    json_filename = f"{tensorboard_dir.parent.name}_data.json"
    if destination is None:
        file_path = tensorboard_dir.parent.joinpath(json_filename)
    else:
        file_path = Path(destination).joinpath(json_filename)

    with open(file_path, "w") as file:
        file.write(json.dumps(parameter_data))

    return file_path


def to_title_case(string: str):
    return " ".join([(x[0].upper() + x[1:].lower()) for x in string.split("_")])


def create_violin_figure(json_file_path, order_by, size_factor=2.85, int_index=False):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    json_file_path = Path(json_file_path)
    with open(json_file_path, "r") as file:
        run_data: Dict[str, Dict[str, Union[float, str, Dict[str, Union[str, float]]]]] = json.loads(file.read())

    max_accuracies: Dict[str, float] = {}
    parameters_map: Dict[str, Dict[str, Union[str, float]]] = {}
    for key, value in run_data.items():
        # key = key[key.find("_") + 1:]
        # if key in max_accuracies:
        #     max_accuracies[key] = max(*value["test_accuracy"].values(), max_accuracies[key])
        # else:
        max_accuracies[key] = max(value["test_accuracy"].values())
        parameters_map[key] = value["parameters"]

    if not (isinstance(order_by, list) or isinstance(order_by, tuple)):
        order_by = (order_by,)

    plot_data = {
        "x": [],
        "y": [],
        "hue": [] if len(order_by) > 1 else None,
        "hue_order": [] if len(order_by) > 1 else None,
    }
    plot_data_m = copy.deepcopy(plot_data)
    plot_data_fm = copy.deepcopy(plot_data)
    for key, value in max_accuracies.items():
        parameters = parameters_map[key]
        if not all([x in parameters for x in order_by]):
            continue

        x_value = parameters[order_by[0]]
        if int_index:
            x_value = float("inf") if x_value == "None" else float(x_value)
        # x_value = "Digital" if x_value == "None" else f"{int(math.log2(int(x_value)))}-bits"
        plot_data["x"].append(x_value)

        if len(order_by) > 1:
            hue = parameters[order_by[1]]
            if int_index:
                hue = float("inf") if hue == "None" else float(hue)
            plot_data["hue"].append(hue)

        plot_data["y"].append(value * 100)
        for k, v in plot_data.items():
            if k == "hue_order":
                continue
            if v is None:
                continue
            if parameters["dataset"] == "MNIST":
                plot_data_m[k].append(v[-1])
            elif parameters["dataset"] == "FashionMNIST":
                plot_data_fm[k].append(v[-1])

    if len(order_by) > 1:
        plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))
        if "Identity" in plot_data["hue_order"]:
            plot_data["hue_order"].remove("Identity")
            plot_data["hue_order"].insert(0, "Identity")
        if "None" in plot_data["hue_order"]:
            plot_data["hue_order"].remove("None")
            plot_data["hue_order"].insert(0, "None")

    plot_data_m["hue_order"] = plot_data["hue_order"]
    plot_data_fm["hue_order"] = plot_data["hue_order"]

    fig = figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)
    # 3.25
    # 1.85
    fig_size = [6.5 / 5, 1.75]
    if isinstance(size_factor, tuple):
        fig_size[0] *= size_factor[0]
        fig_size[1] *= size_factor[1]
    else:
        fig_size[0] *= size_factor
        fig_size[1] *= size_factor

    fig.set_size_inches(*fig_size)
    fig.set_dpi(200)
    hh = 0.6
    color_palette = seaborn.husl_palette(h=hh, l=0.7)
    seaborn.violinplot(**plot_data, cut=0, palette=color_palette, inner=None, linewidth=0.1)
    color_palette = seaborn.husl_palette(h=hh, l=0.45)
    seaborn.stripplot(**plot_data_fm, palette=color_palette, linewidth=0.1, size=3, jitter=1 / 3, dodge=True)
    color_palette = seaborn.husl_palette(h=hh, l=0.65)
    seaborn.stripplot(**plot_data_m, palette=color_palette, linewidth=0.1, size=3, jitter=1 / 3, dodge=True)
    plt.yticks(np.arange(0, 101, 25))
    plt.ylim([0, 100])
    plt.xlabel(to_title_case(order_by[0]).replace(" W", " [W]").replace(" Y", " [Y]"))
    # plt.ylabel("Accuracy")
    if len(order_by) > 1:
        plt.legend(plot_data["hue_order"], title=to_title_case(order_by[1]).replace(" W", " [W]").replace(" Y", " [Y]"))
    # plt.title(json_file_path.name)
    fig.tight_layout()
    plt.show()
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_{"_".join(order_by)}_image.svg',
                dpi=fig.dpi, transparent=True)
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_{"_".join(order_by)}_image.png',
                dpi=fig.dpi, transparent=True)
    close('all')


def create_line_figure(json_file_path, order_by, size_factor=2.85):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    json_file_path = Path(json_file_path)
    with open(json_file_path, "r") as file:
        run_data: Dict[str, Dict[str, Union[float, str, Dict[str, Union[str, float]]]]] = json.loads(file.read())

    max_accuracies: Dict[str, List[float]] = {}
    parameters_map: Dict[str, Dict[str, Union[str, float]]] = {}
    for key, value in run_data.items():
        # key = key[key.find("_") + 1:]
        # if key in max_accuracies:
        #     max_accuracies[key] = max(*value["test_accuracy"].values(), max_accuracies[key])
        # else:
        max_accuracies[key] = value["test_accuracy"].values()
        parameters_map[key] = value["parameters"]

    if not (isinstance(order_by, list) or isinstance(order_by, tuple)):
        order_by = (order_by,)
    plot_data = {
        "x": [],
        "y": [],
        "hue": [],
        # "hue_order": [] if len(order_by) > 1 else None,
    }
    for key, value in max_accuracies.items():
        parameters = parameters_map[key]
        if not all([x in parameters for x in order_by]):
            continue
        # print(parameters)
        for epoch, accuracy in enumerate(np.array(list(value)) * 100):
            plot_data["x"].append(epoch + 1)
            plot_data["y"].append(accuracy)
            plot_data["hue"].append(", ".join([parameters[x] for x in order_by]))

    # if len(order_by) > 1:
    #     plot_data["hue_order"] = list(sorted(set(plot_data["hue"])))
    #     if "Identity" in plot_data["hue_order"]:
    #         plot_data["hue_order"].remove("Identity")
    #         plot_data["hue_order"].insert(0, "Identity")
    #     if "None" in plot_data["hue_order"]:
    #         plot_data["hue_order"].remove("None")
    #         plot_data["hue_order"].insert(0, "None")

    fig = figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)
    # 3.25
    # 1.85
    fig.set_size_inches(*[x * size_factor for x in [1.857, 1.75]])
    fig.set_dpi(200)
    seaborn.lineplot(**plot_data, palette="Set2", linewidth=1.5)
    # plt.yticks(np.arange(0, 101, 20))
    plt.xticks(np.arange(1, 11, 1))
    # plt.ylim([0, 100])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # if len(order_by) > 0:
    #     plt.legend(plot_data["hue_order"], title=to_title_case(order_by).replace(" W", " [W]").replace(" Y", " [Y]"))
    # plt.title(json_file_path.name)
    fig.tight_layout()
    plt.show()
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_{order_by}_image.svg',
                dpi=fig.dpi)
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_{order_by}_image.png',
                dpi=fig.dpi)
    close('all')


def calculate_snr_signal(signal, noise_signal):
    t2 = torch.tensor(2, device=is_cpu_cuda.get_device())
    s = torch.sum(torch.pow(signal, t2))
    n = torch.sum(torch.pow(torch.abs(signal) - torch.abs(noise_signal), t2))
    return s / n


dp_calculate_snr = {
    "0.2_4_MNIST": [11.088152885437012, 8.456015586853027],
    "0.2_16_MNIST": [175.8230438232422, 130.319091796875],
    "0.2_64_MNIST": [2672.276123046875, 1898.3907470703125],
    "0.5_4_MNIST": [3.2126624584198, 2.893296241760254],
    "0.5_16_MNIST": [51.24656295776367, 44.03767013549805],
    "0.5_64_MNIST": [807.6673583984375, 661.8540649414062],
    "0.8_4_MNIST": [0.4601576328277588, 0.6117348670959473],
    "0.8_16_MNIST": [7.358730792999268, 6.856428623199463],
    "0.8_64_MNIST": [117.48218536376953, 104.25345611572266],

    "0.2_4_FashionMNIST": [17.030637741088867, 11.329060554504395],
    "0.2_16_FashionMNIST": [275.156005859375, 166.92759704589844],
    "0.2_64_FashionMNIST": [4364.8115234375, 2605.60400390625],
    "0.5_4_FashionMNIST": [5.585911273956299, 4.426943778991699],
    "0.5_16_FashionMNIST": [89.73049926757812, 60.25444030761719],
    "0.5_64_FashionMNIST": [1431.469482421875, 930.8984985351562],
    "0.8_4_FashionMNIST": [0.8391902446746826, 1.1593579053878784],
    "0.8_16_FashionMNIST": [13.434399604797363, 10.254481315612793],
    "0.8_64_FashionMNIST": [214.82745361328125, 147.53366088867188],

    "0.2_4_Weights": [24.76962661743164, 13.233047485351562],
    "0.5_4_Weights": [9.46220588684082, 5.732192516326904],
    "0.8_4_Weights": [1.7913140058517456, 1.938230037689209],
    "0.2_16_Weights": [370.3149719238281, 188.97683715820312],
    "0.5_16_Weights": [138.59751892089844, 72.18633270263672],
    "0.8_16_Weights": [22.993026733398438, 13.150540351867676],
    "0.2_64_Weights": [5829.53662109375, 2944.764404296875],
    "0.5_64_Weights": [2171.158203125, 1096.6651611328125],
    "0.8_64_Weights": [349.09698486328125, 179.90565490722656],
}


def calculate_snr(leakage, precision, dataset):
    global dp_calculate_snr
    precision = int(precision)
    leakage = float(leakage)
    pp = "_".join([str(x) for x in [leakage, precision, dataset]])
    if pp in dp_calculate_snr:
        return dp_calculate_snr[pp]

    rp = ReducePrecision(precision=precision)
    gn = GaussianNoise(leakage=leakage, precision=precision)
    cl = Clamp01()
    de = is_cpu_cuda.get_device()

    average_response_analog = []
    average_response_digital = []
    if dataset == "MNIST":
        torch_dataset = torchvision.datasets.MNIST
    elif dataset == "FashionMNIST":
        torch_dataset = torchvision.datasets.FashionMNIST
    elif dataset == "Weights":
        cl = Clamp()
        torch_dataset = None
        train_loader = []
        test_loader = []
        weight_distribution = lambda: (nn.init.uniform_(torch.zeros(100, 100), a=-1, b=1))
        for i in range(2000):
            train_loader.append((weight_distribution(), None))
    else:
        raise NotImplemented

    if torch_dataset is not None:
        train_loader, test_loader, input_shape, classes = load_vision_dataset(
            dataset=torch_dataset,
            path="../_data",
            batch_size=128,
            is_cuda=is_cpu_cuda.is_cuda()
        )

    def run_calculate(loader):
        if isinstance(loader, DataLoader):
            # noinspection PyTypeChecker
            dataset_size = len(loader.dataset)
        else:
            dataset_size = len(loader)

        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(de)
            data_ay = gn(rp(cl(data)))
            data_y = rp(cl(gn(data_ay)))
            average_response_analog.append(calculate_snr_signal(data, data_ay))
            average_response_digital.append(calculate_snr_signal(data, data_y))

            if dataset == "Weights":
                print_mod = int(dataset_size / 5)
            else:
                print_mod = int(dataset_size / (len(data) * 5))
            if print_mod > 0 and batch_idx % print_mod == 0 and batch_idx > 0:
                print(
                    f'{pp}:[{batch_idx * len(data)}/{dataset_size} ({100. * batch_idx / len(loader):.0f}%)]: {sum(average_response_analog) / len(average_response_analog):.4f}, {sum(average_response_digital) / len(average_response_digital):.4f}')

    run_calculate(train_loader)
    run_calculate(test_loader)

    average_response_analog = float(sum(average_response_analog) / len(average_response_analog))
    average_response_digital = float(sum(average_response_digital) / len(average_response_digital))
    dp_calculate_snr[pp] = (average_response_analog, average_response_digital)
    print(json.dumps(dp_calculate_snr))
    return dp_calculate_snr[pp]


def create_scatter_figure(json_file_path, size_factor=2.85):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    json_file_path = Path(json_file_path)
    with open(json_file_path, "r") as file:
        run_data: Dict[str, Dict[str, Union[float, str, Dict[str, Union[str, float]]]]] = json.loads(file.read())

    max_accuracies: Dict[str, float] = {}
    parameters_map: Dict[str, Dict[str, Union[str, float]]] = {}
    for key, value in run_data.items():
        # key = key[key.find("_") + 1:]
        # if key in max_accuracies:
        #     max_accuracies[key] = max(*value["test_accuracy"].values(), max_accuracies[key])
        # else:
        max_accuracies[key] = max(value["test_accuracy"].values())
        parameters_map[key] = value["parameters"]

    plot_data = {
        "x": [],
        "y": [],
        "hue": [],
        # "hue_order": [] if len(order_by) > 1 else None,
        "style": [],
    }
    flip = False
    which_snr = 1
    m = 0
    fm = 0
    for key, value in max_accuracies.items():
        parameters = parameters_map[key]
        if not all([x in parameters for x in ["leakage_y", "precision_y", "leakage_w", "precision_w", ]]):
            continue
        # if not (parameters["dataset"] == "FashionMNIST"):
        #     m = max(m, value)
        #     continue
        # fm = max(fm, value)
        # if not (parameters["leakage_w"] == "0.2"):
        #     continue
        # if not (parameters["precision_y"] == "16"):
        #     continue
        # if not (parameters["optimiser_class"] == "StochasticReducePrecisionOptimizer"):
        #     continue
        # if not (parameters["model_class"] == "Linear3"):
        #     continue
        # if not (parameters["activation_class"] == "Tanh"):
        #     continue
        # if not (parameters["activation_class"] == "LeakyReLU"):
        #     continue

        # print(parameters)
        x_value = 10 * math.log10(
            calculate_snr(parameters["leakage_y"], parameters["precision_y"], parameters["dataset"])[1])
        hue = 10 * math.log10(calculate_snr(parameters["leakage_w"], parameters["precision_w"], "Weights")[0])
        plot_data["y"].append(value * 100)
        if flip:
            plot_data["x"].append(x_value)
            plot_data["hue"].append(hue)
        else:
            plot_data["x"].append(hue)
            plot_data["hue"].append(x_value)
        plot_data["style"].append(parameters["dataset"])

    print(m)
    print(fm)
    # if len(order_by) > 1:
    #     plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))
    #     if "Identity" in plot_data["hue_order"]:
    #         plot_data["hue_order"].remove("Identity")
    #         plot_data["hue_order"].insert(0, "Identity")
    #     if "None" in plot_data["hue_order"]:
    #         plot_data["hue_order"].remove("None")
    #         plot_data["hue_order"].insert(0, "None")

    fig = figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)
    # 3.25
    # 1.85
    fig_size = [2, 1.75]
    if isinstance(size_factor, tuple):
        fig_size[0] *= size_factor[0]
        fig_size[1] *= size_factor[1]
    else:
        fig_size[0] *= size_factor
        fig_size[1] *= size_factor

    fig.set_size_inches(*fig_size)
    fig.set_dpi(200)
    # cmap = seaborn.color_palette("flare", len(set(plot_data["hue"])), as_cmap=True)
    cmap = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
    seaborn.lineplot(**plot_data, palette=cmap, linewidth=0.75, ci=1)
    plt.yticks(np.arange(0, 101, 20))
    # plt.xticks(np.arange(0, 41, 7.5))
    plt.ylim([0, 100])
    norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        # ticks=np.round(np.linspace(min(plot_data["hue"]), max(plot_data["hue"]), len(set(plot_data["hue"])))),
    )
    if flip:
        plt.xlabel("SNR [Y] (dB)")
        cbar.ax.set_ylabel("SNR [W] (dB)")
    else:
        plt.xlabel("SNR [W] (dB)")
        cbar.ax.set_ylabel("SNR [Y] (dB)")

    fig.tight_layout()
    plt.show()
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_snr_image.svg',
                dpi=fig.dpi, transparent=True)
    fig.savefig(f'_data/{timestamp}_{json_file_path.name.replace(".json", "")}_snr_image.png',
                dpi=fig.dpi, transparent=True)
    close('all')


if __name__ == '__main__':
    data_folder = Path("C:/_data")
    timestamp = int(time.time())

    norm_order_by = ("norm_class_w", "norm_class_y")
    precision_order_by = ("precision_w", "precision_y")
    leakage_order_by = ("leakage_w", "leakage_y")

    # for i in os.listdir(data_folder):
    #     collect_parameters_to_json(data_folder.joinpath(i), data_folder)

    # aa = 1.5
    # for i in [
    #     ("tensorboard_cleo_run_1_data.json", "norm_class_w", (aa*2, aa/1.75)),
    #     ("tensorboard_cleo_run_3_data.json", "leakage_w", (aa*2, aa/1.75)),
    #
    #     ("tensorboard_cleo_run_2_data.json", ("precision_w", "precision_class"), (aa*1.75, aa*1.25)),
    #     # ("tensorboard_cleo_run_2S_F_data.json", precision_order_by, (aa*2, aa/2), True),
    #
    #     # ("tensorboard_cleo_run_3_data.json", ("leakage_w", "precision_w"), (4, 2), True),
    #
    #     # ("tensorboard_cleo_run_3_data.json", "activation_class"),
    # ]:
    #     create_violin_figure(data_folder.joinpath(i[0]), *i[1:])

    # create_line_figure(data_folder.joinpath("tensorboard_cleo_run_3_data.json"), ("leakage_w", "precision_w"))
    aa = 1.75
    create_scatter_figure(data_folder.joinpath("tensorboard_cleo_run_3_data.json"), (1 * aa, 1 * aa))

    # calculate_snr(0.2, 4, "Weights")
    # calculate_snr(0.5, 4, "Weights")
    # calculate_snr(0.8, 4, "Weights")
    # calculate_snr(0.2, 16, "Weights")
    # calculate_snr(0.5, 16, "Weights")
    # calculate_snr(0.8, 16, "Weights")
    # calculate_snr(0.2, 64, "Weights")
    # calculate_snr(0.5, 64, "Weights")
    # calculate_snr(0.8, 64, "Weights")
    # calculate_snr(0.2, 4, "MNIST")
    # calculate_snr(0.5, 4, "MNIST")
    # calculate_snr(0.8, 4, "MNIST")
    # calculate_snr(0.2, 16, "MNIST")
    # calculate_snr(0.5, 16, "MNIST")
    # calculate_snr(0.8, 16, "MNIST")
    # calculate_snr(0.2, 64, "MNIST")
    # calculate_snr(0.5, 64, "MNIST")
    # calculate_snr(0.8, 64, "MNIST")
    # calculate_snr(0.2, 4, "FashionMNIST")
    # calculate_snr(0.5, 4, "FashionMNIST")
    # calculate_snr(0.8, 4, "FashionMNIST")
    # calculate_snr(0.2, 16, "FashionMNIST")
    # calculate_snr(0.5, 16, "FashionMNIST")
    # calculate_snr(0.8, 16, "FashionMNIST")
    # calculate_snr(0.2, 64, "FashionMNIST")
    # calculate_snr(0.5, 64, "FashionMNIST")
    # calculate_snr(0.8, 64, "FashionMNIST")
