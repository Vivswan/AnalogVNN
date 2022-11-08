import itertools
import json
import math
import os
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.colors
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from seaborn.palettes import _color_to_rgb, _ColorPalette

from nn.layers.noise.GaussianNoise import GaussianNoise


def main_color_palette(n_colors=6, as_cmap=False):  # noqa
    if as_cmap:
        n_colors = 256

    hues = np.linspace(130, -115, int(n_colors)) % 360
    saturation = np.linspace(1, 1, int(n_colors)) * 99
    lightness = np.linspace(0.85, 0.3, int(n_colors)) * 99

    palette = [
        _color_to_rgb((h_i, s_i, l_i), input="husl")
        for h_i, s_i, l_i in zip(hues, saturation, lightness)
    ]
    palette = list(reversed(palette))
    if as_cmap:
        return matplotlib.colors.ListedColormap(palette, "hsl")
    else:
        return _ColorPalette(palette)


def to_title_case(string: str):
    string = string.split(".")[-1]
    string = [(x[0].upper() + x[1:].lower()) for x in string.split("_")]
    string = " ".join(string)
    if string.split(" ")[0] == "Std":
        string = " ".join(["σ", *string.split(" ")[1:]])
    string = string.replace(" W", " [W]").replace(" Y", " [Y]")
    return string.replace('Leakage', 'Error Probability')


def apply_if_not_none(value, func):
    if value is None:
        return value
    return func(value)


def sanitise_data(data):
    data["train_loss"] = data["loss_accuracy"]["train_loss"][-1] * 100
    data["train_accuracy"] = data["loss_accuracy"]["train_accuracy"][-1] * 100
    data["test_loss"] = data["loss_accuracy"]["test_loss"][-1] * 100
    data["test_accuracy"] = data["loss_accuracy"]["test_accuracy"][-1] * 100

    data["###"] = int(f"{abs(hash(data['parameter_log']['dataset']))}"[:2]) / 100

    py = data["hyperparameters_nn_model"]["precision_y"]
    pw = data["hyperparameters_weight_model"]["precision_w"]
    data["bit_precision_y"] = 32.0 if py is None else math.log2(py)
    data["bit_precision_w"] = 32.0 if pw is None else math.log2(pw)

    if "num_linear_layer" in data["parameter_log"]:
        data["parameter_log"]["num_linear_layer"] += 1
    if "num_linear_layer" in data["hyperparameters_nn_model"]:
        data["hyperparameters_nn_model"]["num_linear_layer"] += 1

    if data["parameter_log"]["precision_class_w"] == 'None':
        data["parameter_log"]["precision_class_w"] = "Digital"
    if data["parameter_log"]["precision_class_y"] == 'None':
        data["parameter_log"]["precision_class_y"] = "Digital"

    if data["parameter_log"]["precision_y"] is not None \
            and data["parameter_log"]["leakage_y"] is not None:
        data["std_y"] = GaussianNoise.calc_std(
            data["parameter_log"]["leakage_y"],
            data["parameter_log"]["precision_y"]
        )

    if data["parameter_log"]["precision_w"] is not None \
            and data["parameter_log"]["leakage_w"] is not None:
        data["std_w"] = GaussianNoise.calc_std(
            data["parameter_log"]["leakage_w"],
            data["parameter_log"]["precision_w"]
        )

    return data


def get_combined_data(data_path):
    data_path = Path(data_path)
    if data_path.is_file():
        with open(data_path, "r") as file:
            data = json.loads(file.read())
        return data

    data = {}

    for i in os.listdir(data_path):
        with open(data_path.joinpath(str(i)), "r") as file:
            dd = json.loads(file.read())
            data[str(i)] = sanitise_data(dd)
        with open(data_path.joinpath(str(i)), "w") as file:
            file.write(json.dumps(dd, indent=2, sort_keys=True))

    return data


def compile_data(data_path):
    data_path = Path(data_path)
    run_data = get_combined_data(data_path)
    with open(f"{data_path}.json", "w") as file:
        file.write(json.dumps(run_data, indent=2))


def get_key(obj, key):
    key = key.split(".")
    for i in key:
        obj = obj[i]
    return obj


def get_filtered_data(data, filters):
    if filters is None or len(filters.keys()) == 0:
        return data

    filtered_data = {}

    for key, value in data.items():
        check = True
        for filter_key, filter_value in filters.items():
            if isinstance(filter_value, str):
                if not any([get_key(data[key], filter_key) == i for i in filter_value.split("|")]):
                    check = False
                    break
            else:
                if get_key(data[key], filter_key) != filter_value:
                    check = False
                    break

        if check:
            filtered_data[key] = value

    return filtered_data


def get_plot_data(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None, add_data=None):
    if add_data is None:
        add_data = {}

    plot_labels = {}
    plot_data = {}
    add_data["x"] = x_axis
    add_data["y"] = y_axis
    add_data["hue"] = subsection
    add_data["style"] = colorbar

    if isinstance(data_path, list):
        run_data = {}
        for i in data_path:
            data = get_combined_data(Path(i))
            run_data = {**run_data, **data}
    else:
        run_data = get_combined_data(Path(data_path))

    run_data = get_filtered_data(run_data, filters)

    for key, value in add_data.items():
        if value is None:
            continue

        plot_labels[key] = value
        plot_data[key] = []

    for key, value in run_data.items():
        for i in plot_labels:
            plot_data[i].append(get_key(run_data[key], plot_labels[i]))

    if colorbar is None:
        if subsection is not None:
            plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))
            if "Digital" in plot_data["hue_order"]:
                plot_data["hue_order"].remove("Digital")
                plot_data["hue_order"].insert(0, "Digital")
            if "None" in plot_data["hue_order"]:
                plot_data["hue_order"].remove("None")
                plot_data["hue_order"].insert(0, "None")
    else:
        if "hue" not in plot_data:
            plot_data["hue"] = plot_data["style"]
            del plot_data["style"]
        else:
            plot_data["hue"], plot_data["style"] = plot_data["style"], plot_data["hue"]

    zip_list = ["x", "y"]
    if "hue" in plot_data:
        zip_list.append("hue")
    if "style" in plot_data:
        zip_list.append("style")

    if isinstance(plot_data["x"][0], str):
        ziped_list = list(zip(*[plot_data[x] for x in zip_list]))
        ziped_list = sorted(ziped_list, key=lambda tup: -np.sum(np.array(tup[0]) == "None"))
        unziped_list = list(zip(*ziped_list))

        for i, v in enumerate(zip_list):
            plot_data[v] = list(unziped_list[i])
    return plot_data


def pick_max_from_plot_data(plot_data, max_keys, max_value):
    max_keys_value = []
    for i in max_keys:
        max_keys_value.append(list(set(plot_data[i])))

    max_accuracies = {i: 0 for i in list(itertools.product(*max_keys_value))}
    for index, value in enumerate(plot_data[max_value]):
        index_max_key_values = tuple([plot_data[i][index] for i in max_keys])

        if max_accuracies[index_max_key_values] < value:
            max_accuracies[index_max_key_values] = value

    plot_data[max_value] = []
    for i in max_keys:
        plot_data[i] = []

    max_accuracies = sorted(max_accuracies.items(), key=lambda tup: tup[0])
    max_accuracies = sorted(max_accuracies, key=lambda tup: -np.sum(np.array(tup[0]) == "None"))

    for key, value in max_accuracies:
        for index, val in enumerate(max_keys):
            plot_data[max_keys[index]].append(key[index])

        plot_data[max_value].append(value)
    return plot_data


def pre_plot(size_factor):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)

    # fig_size = [3.25, 1.85]
    # fig_size = [2.00, 1.75]
    fig_size = 1.75

    fig_size = tuple((np.array(fig_size) * np.array(size_factor)).tolist())
    fig.set_size_inches(*fig_size)
    fig.set_dpi(200)

    return fig


def post_plot(plot_data):
    x_axis_title = to_title_case(plot_data["x_axis"])
    y_axis_title = to_title_case(plot_data["y_axis"])
    filter_text = ""

    if plot_data["filters"] is not None:
        filter_text = " {" + "-".join(
            [f"{to_title_case(key)}={value}" for key, value in plot_data["filters"].items()]) + "}"
        filter_text = filter_text.replace("|", " or ")
        # plt.title(f"Filters = {filter_text}")

    plt.yticks(np.arange(0, 101, 25))
    plt.ylim([0, 100])
    plt.xlabel(x_axis_title)
    # plt.ylabel(y_axis_title)
    plt.ylabel((plot_data["y_prefix"] if "y_prefix" in plot_data else "") + y_axis_title)

    if plot_data["subsection"] is not None:
        if "g" in plot_data:
            h, l = plot_data["g"].get_legend_handles_labels()

            if plot_data["colorbar"] is None:
                subsection_len = len(set(plot_data["hue"]))
            else:
                subsection_len = len(set(plot_data["style"]))

            plt.legend(h[-subsection_len:], l[-subsection_len:], title=to_title_case(plot_data["subsection"]))
        else:
            plt.legend(title=to_title_case(plot_data["subsection"]))
    elif plot_data["colorbar"] is not None:
        plt.legend(title=to_title_case(plot_data["colorbar"]))

    plot_data["fig"].tight_layout()
    # plt.show()

    if isinstance(plot_data["data_path"], list):
        run_name = "".join([Path(i).name for i in plot_data["data_path"]])
    else:
        run_name = Path(plot_data["data_path"]).name[:Path(plot_data["data_path"]).name.index(".")]

    subsection_text = "" if plot_data["subsection"] is None else f" #{to_title_case(plot_data['subsection'])}"
    colorbar_text = "" if plot_data["colorbar"] is None else f" #{to_title_case(plot_data['colorbar'])}"

    name = f"{plot_data['prefix']} - {run_name} - {x_axis_title} vs {y_axis_title}{filter_text}{colorbar_text}{subsection_text}"

    plot_data["fig"].savefig(f'{location}/{name}.svg', dpi=plot_data["fig"].dpi, transparent=True)
    plot_data["fig"].savefig(f'{location}/{name}.png', dpi=plot_data["fig"].dpi, transparent=True)

    plt.close('all')


def create_violin_figure(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                         size_factor: Tuple[float, float] = 2, color_by=None):
    if filters is None:
        filters = {}
    if colorbar is not None:
        subsection = colorbar

    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, filters=filters,
                              add_data={"color_by": color_by})

    fig = pre_plot(size_factor)
    # color_by_data = None
    # if color_by is not None:
    #     color_by_data = plot_data["color_by"]
    #     del plot_data["color_by"]

    n_colors = None
    n_colors = len(plot_data["hue_order"]) if ("hue" in plot_data and n_colors is None) else n_colors
    n_colors = len(set(plot_data["x"])) if n_colors is None else n_colors
    color_map = main_color_palette(n_colors=n_colors)
    g = seaborn.violinplot(**plot_data, cut=0, palette=color_map, inner=None, linewidth=0.1)
    color_map = main_color_palette(n_colors=n_colors)

    # if color_by is not None:
    #     for i, color_by_value in enumerate(set(color_by_data)):
    #         new_plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, filters={
    #             **filters,
    #             color_by: color_by_value
    #         })
    #         gs = seaborn.stripplot(**new_plot_data, palette=[color_map[len(set(plot_data["x"])) + i]], linewidth=0.1, size=3, jitter=1 / 10, dodge=True)
    # else:
    #     gs = seaborn.stripplot(**plot_data, palette=color_map, linewidth=0.1, size=3, jitter=1 / 10, dodge=True)
    gs = seaborn.stripplot(**plot_data, palette=color_map, linewidth=0.1, size=3, jitter=1 / 10, dodge=True)
    if colorbar is not None:
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))
    # if color_by is not None:
    #     color_by_data, plot_data["hue"] = plot_data["hue"], color_by_data
    #     plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))

    # if color_by is not None:
    #     color_map = seaborn.color_palette("flare", len(set(plot_data["hue"])), as_cmap=True)
    #     # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
    #     norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
    #     scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    #     scalar_map.set_array([])
    #     cbar = fig.colorbar(scalar_map, ax=gs)
    #     cbar.ax.set_title("\"bins\"")

    # if color_by is not None:
    #     color_by_data, plot_data["hue"] = plot_data["hue"], color_by_data
    #     plot_data["hue_order"] = sorted(list(set(plot_data["hue"])))

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "v"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["gs"] = gs
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = None
    plot_data["filters"] = filters
    post_plot(plot_data)


def create_line_figure(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                       size_factor: Tuple[float, float] = 2, ci=1):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)

    fig = pre_plot(size_factor)

    color_map = main_color_palette(n_colors=10)
    if colorbar is not None:
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))

    g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, ci=ci)

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "l"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = colorbar
    plot_data["filters"] = filters
    plot_data["y_prefix"] = "Average "
    post_plot(plot_data)


def create_line_figure_max(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None,
                           size_factor: Tuple[float, float] = 2.0, x_lim=None):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)
    fig = pre_plot(size_factor)

    max_keys = ["x"]

    if subsection is not None:
        max_keys.append("hue")
    if colorbar is not None and subsection is not None:
        max_keys.append("style")
    if colorbar is not None and subsection is None:
        max_keys.append("hue")

    plot_data = pick_max_from_plot_data(plot_data, max_keys, "y")

    if colorbar is not None:
        color_map = main_color_palette(n_colors=256)
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("hsl", color_map)
        # color_map = seaborn.color_palette("cubehelix", len(set(plot_data["hue"])), as_cmap=True)
        # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))
        g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, errorbar=('ci', 1), hue_norm=norm)
    else:
        color_map = main_color_palette(n_colors=len(plot_data["hue_order"]))
        g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, errorbar=('ci', 1))

    if x_lim is not None:
        g.set_xlim(*x_lim)
    # g.set(yscale="log")

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "lm"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = colorbar
    plot_data["filters"] = filters
    plot_data["y_prefix"] = "Maximum "
    post_plot(plot_data)


def calculate_max_accuracy(data_path, test_in):
    data_path = Path(data_path)
    plot_data = get_plot_data(data_path, test_in, "loss_accuracy.test_accuracy")
    max_accuracies = {}
    for i in set(plot_data["x"]):
        max_accuracies[i] = 0.0

    for index, value in enumerate(plot_data["y"]):
        value = max(value)
        if max_accuracies[plot_data["x"][index]] < value:
            max_accuracies[plot_data["x"][index]] = value
            max_accuracies[plot_data["x"][index] + "_index"] = index

    print(max_accuracies)


def create_analogvnn1_figures_va1():
    create_line_figure_max(
        f"{location}/analogvnn_1.2_json.json",
        "parameter_log.num_linear_layer",
        "test_accuracy",
        colorbar="parameter_log.num_conv_layer",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
        filters={
            "parameter_log.norm_class_w": "None",
            "parameter_log.norm_class_y": "None",
        }
    )
    create_violin_figure(
        f"{location}/analog_vnn_3.json",
        "parameter_log.activation_class",
        "test_accuracy",
        size_factor=(6.5 * 2 / 3, 1.61803398874),
        subsection="parameter_log.dataset",
    )


def create_analogvnn1_figures_va2():
    create_line_figure_max(
        f"{location}/analogvnn_1.2_json.json",
        "parameter_log.norm_class_w",
        "test_accuracy",
        subsection="parameter_log.norm_class_y",
        size_factor=(6.5, 2),
    )
    create_violin_figure(
        f"{location}/analogvnn_1.2_json.json",
        "parameter_log.norm_class_w",
        "test_accuracy",
        size_factor=(6.5, 2),
        subsection="parameter_log.dataset",
    )


def create_analogvnn1_figures_va3():
    create_violin_figure(
        f"{location}/analog_vnn_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
        color_by="bit_precision_w",
        filters={
            "parameter_log.dataset": "MNIST",
            "parameter_log.norm_class_w": "Clamp",
            "parameter_log.norm_class_y": "Clamp",
        },
    )
    create_violin_figure(
        f"{location}/analog_vnn_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
        color_by="bit_precision_w",
        filters={
            "parameter_log.dataset": "FashionMNIST",
            "parameter_log.norm_class_w": "Clamp",
            "parameter_log.norm_class_y": "Clamp",
        },
    )
    create_violin_figure(
        f"{location}/analog_vnn_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
        color_by="bit_precision_w",
        filters={
            "parameter_log.dataset": "CIFAR10",
            "parameter_log.norm_class_w": "Clamp",
            "parameter_log.norm_class_y": "Clamp",
        },
    )
    create_line_figure(
        f"{location}/analog_vnn_3.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="bit_precision_y",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )
    create_line_figure_max(
        f"{location}/analog_vnn_3.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="bit_precision_y",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )


def create_analogvnn1_figures_va4():
    create_violin_figure(
        f"{location}/analog_vnn_3.json",
        "parameter_log.dataset",
        "test_accuracy",
        subsection="parameter_log.leakage_w",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
    )
    create_line_figure(
        f"{location}/analog_vnn_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        colorbar="parameter_log.leakage_y",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
    )
    create_line_figure_max(
        f"{location}/analog_vnn_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        colorbar="parameter_log.leakage_y",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
    )
    create_line_figure_max(
        f"{location}/analog_vnn_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        colorbar="bit_precision_w",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )
    create_line_figure_max(
        f"{location}/analog_vnn_3.json",
        "std_w",
        "test_accuracy",
        colorbar="std_y",
        subsection="parameter_log.dataset",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )


def create_parneet_figures_vb1():
    create_line_figure(
        f"{location}/runs_parneet_1_b_json.json",
        "parameter_log.batch_size",
        "test_accuracy",
        ci="sd",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )
    create_line_figure(
        f"{location}/runs_parneet_1_b_json.json",
        "parameter_log.activation_class",
        "test_accuracy",
        colorbar="parameter_log.batch_size",
        size_factor=(6.5 * 1 / 2, 1.61803398874 * 1.2),
    )
    create_line_figure_max(
        f"{location}/runs_parneet_2_n3_json.json",
        "parameter_log.norm_class_w",
        "test_accuracy",
        subsection="parameter_log.norm_class_y",
        size_factor=(6.5, 1.61803398874),
    )
    # create_violin_figure(
    #     f"{location}/runs_parneet_2_n3_json.json",
    #     "parameter_log.norm_class_w",
    #     "test_accuracy",
    #     size_factor=(6.5, 1.61803398874 * 1.2),
    # )


def create_parneet_figures_vb2():
    create_line_figure_max(
        f"{location}/runs_parneet_3_p_json.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="bit_precision_y",
        size_factor=(6.5 / 2, 1.61803398874 * 1.2),
    )
    create_line_figure_max(
        f"{location}/runs_parneet_3_p_json.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.activation_class",
        size_factor=(6.5 / 2, 1.61803398874 * 1.2),
    )


def create_parneet_figures_vb3():
    # create_violin_figure(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "parameter_log.color",
    #     "test_accuracy",
    #     colorbar="bit_precision_w",
    #     size_factor=(6.5 * 1/3, 1.61803398874),
    # )
    # create_line_figure_max(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     subsection="parameter_log.activation_class",
    #     size_factor=(6.5 * 1/3, 1.61803398874),
    # )
    # create_line_figure_max(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "bit_precision_w",
    #     "test_accuracy",
    #     colorbar="bit_precision_y",
    #     size_factor=(6.5 * 1/3, 1.61803398874),
    # )

    create_line_figure_max(
        f"{location}/runs_parneet_4_g_json.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        colorbar="parameter_log.leakage_y",
        size_factor=(6.5 * 1 / 3, 1.61803398874),
    )
    # create_line_figure_max(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     colorbar="bit_precision_w",
    #     size_factor=(6.5 * 1/3, 1.61803398874),
    # )
    # create_line_figure_max(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "std_w",
    #     "test_accuracy",
    #     colorbar="std_y",
    #     x_lim=[0, 0.08],
    #     size_factor=(6.5 * 1/3, 1.61803398874),
    # )


def create_analogvnn1_figures():
    create_analogvnn1_figures_va1()
    create_analogvnn1_figures_va2()
    # create_analogvnn1_figures_va3()
    create_analogvnn1_figures_va4()


def create_parneet_figures():
    create_parneet_figures_vb1()
    # create_parneet_figures_vb2()
    # create_parneet_figures_vb3()


if __name__ == '__main__':
    location = "C:/_data"
    # compile_data(f"{location}/analog_vnn_1")
    # compile_data(f"{location}/analog_vnn_2")
    # compile_data(f"{location}/analog_vnn_3")
    # compile_data(f"{location}/analogvnn_1.2_json")

    # compile_data(f"{location}/runs_parneet_1_b_json")
    # compile_data(f"{location}/runs_parneet_2_n3_json")
    # compile_data(f"{location}/runs_parneet_2_n_json")
    # compile_data(f"{location}/runs_parneet_3_p_json")
    # compile_data(f"{location}/runs_parneet_4_g_json")

    # create_analogvnn1_figures()
    # create_parneet_figures()

    create_line_figure_max(
        f"{location}/runs_parneet_4_g_json.json",
        "std_w",
        "test_accuracy",
        colorbar="std_y",
        x_lim=[0, 0.08],
        size_factor=(6.5 * 1 / 3, 1.61803398874),
    )

    # data = get_combined_data(Path(f"{location}/analogvnn_1.2_json.json"))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":0})
    # print("MNIST 0,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":0})
    # print("MNIST 0,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":0})
    # print("MNIST 0,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":3})
    # print("MNIST 3,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":3})
    # print("MNIST 3,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":3})
    # print("MNIST 3,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    #
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":0})
    # print("FashionMNIST 0,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":0})
    # print("FashionMNIST 0,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":0})
    # print("FashionMNIST 0,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":3})
    # print("FashionMNIST 3,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":3})
    # print("FashionMNIST 3,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.FashionMNIST.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":3})
    # print("FashionMNIST 3,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    #
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":0})
    # print("CIFAR10 0,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":0})
    # print("CIFAR10 0,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":0})
    # print("CIFAR10 0,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":1, "parameter_log.num_conv_layer":3})
    # print("CIFAR10 3,1 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":2, "parameter_log.num_conv_layer":3})
    # print("CIFAR10 3,2 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.CIFAR10.__name__, "parameter_log.num_linear_layer":3, "parameter_log.num_conv_layer":3})
    # print("CIFAR10 3,3 max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))

    # data = get_combined_data(Path(f"{location}/analog_vnn_3.json"))
    # filtered_data = get_filtered_data(data, {"parameter_log.dataset": torchvision.datasets.MNIST.__name__,
    #                                          "bit_precision_w": 4, "bit_precision_y": 2})
    # print("MNIST max: ", np.max([get_key(data[key], "test_accuracy") for key, value in filtered_data.items()]))

    # create_line_figure_max(
    #     f"{location}/runs_parneet_4_g_json.json",
    #     "std_w",
    #     "test_accuracy",
    #     colorbar="std_y",
    #     size_factor=(3.25, 1.5),
    # )
    # colormaps = ["rocket", "mako", "flare", "crest", "magma", "viridis", "rocket_r", "cubehelix", "seagreen", "dark:salmon_r", "YlOrBr", "Blues", "vlag", "icefire", "coolwarm", "Spectral"]
    # colormaps_dict = {}
    # for i in colormaps:
    #     try:
    #         colormaps_dict["cm_" + i] = seaborn.color_palette(i, n_colors=256, as_cmap=True).colors
    #     except Exception as e:
    #         print(i)
    # scipy.io.savemat("seaborn.colormap.mat", colormaps_dict)
