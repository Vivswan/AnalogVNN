import itertools
import json
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import seaborn
from matplotlib import pyplot as plt

from nn.layers.noise.GaussianNoise import GaussianNoise


def to_title_case(string: str):
    string = string.split(".")[-1]
    string = [(x[0].upper() + x[1:].lower()) for x in string.split("_")]
    string = " ".join(string)
    if string.split(" ")[0] == "Std":
        string = " ".join(["σ", *string.split(" ")[1:]])
    string = string.replace(" W", " [W]").replace(" Y", " [Y]")
    return string


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

    data["parameter_log"]["num_linear_layer"] += 1
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
            data[str(i)] = sanitise_data(json.loads(file.read()))

    return data


def compile_data(data_path):
    data_path = Path(data_path)
    run_data = get_combined_data(data_path)
    with open(f"{data_path}.json", "w") as file:
        file.write(json.dumps(run_data))


def get_key(obj, key):
    key = key.split(".")
    for i in key:
        obj = obj[i]
    return obj


def get_filtered_data(data, filters):
    if filters is None:
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


def get_plot_data(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None):
    if isinstance(data_path, list):
        run_data = {}
        for i in data_path:
            data = get_combined_data(Path(i))
            run_data = {**run_data, **data}
    else:
        run_data = get_combined_data(Path(data_path))

    run_data = get_filtered_data(run_data, filters)
    x_data = []
    y_data = []
    hue_data = None if subsection is None else []
    style_data = None if colorbar is None else []
    for key, value in run_data.items():
        x_data.append(get_key(run_data[key], x_axis))
        y_data.append(get_key(run_data[key], y_axis))
        if subsection is not None:
            hue_data.append(get_key(run_data[key], subsection))
        if colorbar is not None:
            style_data.append(get_key(run_data[key], colorbar))

    if colorbar is None:
        hue_order = None if subsection is None else sorted(list(set(hue_data)))
    else:
        hue_order = None
        hue_data, style_data = style_data, hue_data

    plot_data = {
        "x": x_data,
        "y": y_data,
    }

    if hue_data is not None:
        plot_data["hue"] = hue_data
        plot_data["hue_order"] = hue_order
        # plot_data["hue_order"] = list(reversed(hue_order))
    if style_data is not None:
        plot_data["style"] = style_data

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

    for key, value in max_accuracies.items():
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
    fig_size = [2.00, 1.75]

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

    plot_data["fig"].tight_layout()
    plt.show()

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


def create_violin_figure(data_path, x_axis, y_axis, subsection=None, filters=None, size_factor=2):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, filters=filters)

    fig = pre_plot(size_factor)

    hh = 0.6
    color_palette = seaborn.husl_palette(h=hh, l=0.7)
    g = seaborn.violinplot(**plot_data, cut=0, palette=color_palette, inner=None, linewidth=0.1)
    color_palette = seaborn.husl_palette(h=hh, l=0.65)
    seaborn.stripplot(**plot_data, palette=color_palette, linewidth=0.1, size=3, jitter=1 / 10, dodge=True)

    plot_data["data_path"] = data_path
    plot_data["prefix"] = "v"
    plot_data["fig"] = fig
    plot_data["g"] = g
    plot_data["x_axis"] = x_axis
    plot_data["y_axis"] = y_axis
    plot_data["subsection"] = subsection
    plot_data["colorbar"] = None
    plot_data["filters"] = filters
    post_plot(plot_data)


def create_line_figure(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None, size_factor=2):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)

    fig = pre_plot(size_factor)

    color_map = "Set2"
    if colorbar is not None:
        color_map = seaborn.color_palette("flare", len(set(plot_data["hue"])), as_cmap=True)
        # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))

    g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, ci=1)

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


def create_line_figure_max(data_path, x_axis, y_axis, subsection=None, colorbar=None, filters=None, size_factor=2):
    plot_data = get_plot_data(data_path, x_axis, y_axis, subsection=subsection, colorbar=colorbar, filters=filters)
    fig = pre_plot(size_factor)

    max_keys = ["x"]

    if subsection is not None:
        max_keys.append("hue")
    if colorbar is not None:
        max_keys.append("style")

    plot_data = pick_max_from_plot_data(plot_data, max_keys, "y")

    color_map = "Set2"
    if colorbar is not None:
        color_map = seaborn.color_palette("flare", len(set(plot_data["hue"])), as_cmap=True)
        # color_map = seaborn.dark_palette("#69d", n_colors=len(set(plot_data["hue"])), reverse=True, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=min(plot_data["hue"]), vmax=max(plot_data["hue"]))
        scalar_map = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(scalar_map)
        cbar.ax.set_ylabel(to_title_case(colorbar))

    g = seaborn.lineplot(**plot_data, palette=color_map, linewidth=1, ci=1)
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


def create_runs_violin_figures():
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.norm_class_w", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.norm_class_y", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.dataset", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.approach", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.activation_class", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.num_linear_layer", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.num_conv_layer", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_2.json", "parameter_log.precision_class_y", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_2.json", "parameter_log.precision_class_w", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_2.json", "bit_precision_y", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_2.json", "bit_precision_w", "test_accuracy")
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )
    create_violin_figure(f"{location}/analog_vnn_run_3.json", "bit_precision_y", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_3.json", "bit_precision_w", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_3.json", "parameter_log.leakage_y", "test_accuracy")
    create_violin_figure(f"{location}/analog_vnn_run_3.json", "parameter_log.leakage_w", "test_accuracy")
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_y",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_w",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
        filters={"parameter_log.dataset": "MNIST"},
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_y",
        "test_accuracy",
        subsection="parameter_log.precision_class_y",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        subsection="parameter_log.precision_class_w",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_y",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )
    create_violin_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        subsection="parameter_log.dataset",
    )


def create_runs_line_figures():
    create_line_figure(
        f"{location}/analog_vnn_run_2.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="bit_precision_y",
        subsection="parameter_log.dataset",
    )
    create_line_figure(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="bit_precision_y",
        subsection="parameter_log.dataset",
    )
    create_line_figure(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.leakage_w",
        "test_accuracy",
        colorbar="parameter_log.leakage_y",
        subsection="parameter_log.dataset",
    )
    create_line_figure(
        f"{location}/analog_vnn_run_3.json",
        "std_w",
        "test_accuracy",
        colorbar="std_y",
        subsection="parameter_log.dataset",
    )


if __name__ == '__main__':
    location = "C:/_data/_json"
    # compile_data(f"{location}/analog_vnn_run_1")
    # compile_data(f"{location}/analog_vnn_run_2")
    # compile_data(f"{location}/analog_vnn_run_3")
    # create_runs_violin_figures()
    # create_runs_line_figures()
    # create_violin_figure(f"{location}/analog_vnn_run_1.json", "parameter_log.norm_class_w", "test_accuracy", size_factor=(3, 1.5))
    # create_violin_figure(
    #     f"{location}/analog_vnn_run_3.json",
    #     "parameter_log.dataset",
    #     "test_accuracy",
    #     subsection="parameter_log.leakage_w",
    #     size_factor=(3, 1.5),
    # )
    # create_violin_figure(
    #     f"{location}/analog_vnn_run_3.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     subsection="parameter_log.dataset",
    #     size_factor=(3, 1.5),
    #     # filters={
    #     #     "parameter_log.dataset": "MNIST",
    #     #     "parameter_log.leakage_y": 0.25,
    #     #     "parameter_log.leakage_w": 0.25,
    #     # },
    # )
    # create_line_figure(
    #     f"{location}/analog_vnn_run_3.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2, 1.25),
    # )
    # create_violin_figure(
    #     [f"{location}/analog_vnn_run_2.json", f"{location}/analog_vnn_run_1.json"],
    #     "bit_precision_w",
    #     "test_accuracy",
    #     subsection="parameter_log.precision_class_w",
    #     filters={
    #         "parameter_log.dataset": "MNIST|FashionMNIST",
    #         "parameter_log.norm_class_y": "Clamp",
    #         "parameter_log.norm_class_w": "Clamp",
    #         "parameter_log.approach": "default",
    #     },
    # )
    # create_line_figure(
    #     f"{location}/analog_vnn_run_3.json",
    #     "std_w",
    #     "test_accuracy",
    #     colorbar="std_y",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2.5, 2)
    # )
    # create_line_figure_max(
    #     f"{location}/analog_vnn_run_3.json",
    #     "std_w",
    #     "test_accuracy",
    #     colorbar="std_y",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2.5, 2)
    # )

    # create_line_figure_max(
    #     f"{location}/analog_vnn_run_2.json",
    #     "bit_precision_w",
    #     "test_accuracy",
    #     # colorbar="std_y",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2.5, 2)
    # )

    create_line_figure_max(
        f"{location}/analog_vnn_run_3.json",
        "parameter_log.num_linear_layer",
        "test_accuracy",
        colorbar="parameter_log.num_conv_layer",
        subsection="parameter_log.dataset",
        size_factor=(2.5, 2),
        filters={
            "bit_precision_w": 6.0,
            "bit_precision_y": 6.0,
        }
    )

    create_line_figure_max(
        f"{location}/analog_vnn_run_3.json",
        "bit_precision_w",
        "test_accuracy",
        colorbar="parameter_log.num_linear_layer",
        subsection="parameter_log.dataset",
        size_factor=(2.5, 2),
        filters={
            "parameter_log.num_conv_layer": 0
        }
    )

    # create_line_figure_max(
    #     f"{location}/analog_vnn_run_3.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     colorbar="parameter_log.leakage_y",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2.5, 2),
    #     filters={
    #         "bit_precision_y": 2.0,
    #         "bit_precision_w": 2.0,
    #     }
    # )
    # create_line_figure_max(
    #     f"{location}/analog_vnn_run_3.json",
    #     "parameter_log.leakage_w",
    #     "test_accuracy",
    #     colorbar="bit_precision_w",
    #     subsection="parameter_log.dataset",
    #     size_factor=(2.5, 2),
    # )
    # calculate_max_accuracy(f"{location}/analog_vnn_run_3.json", "parameter_log.dataset")
