import argparse
import json
import os
from pathlib import Path
from typing import List

import torch


def data_from_tensorboard(tensorboard_dir, destination: Path = None):
    from tensorboard.plugins.hparams.metadata import SESSION_START_INFO_TAG
    from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
    from tensorflow.python.summary.summary_iterator import summary_iterator

    tensorboard_dir = Path(tensorboard_dir)
    all_files: List[Path] = []
    parameter_data = {}

    for root, dirs, files in os.walk(tensorboard_dir):
        for file in files:
            all_files.append(Path(root).joinpath(file))

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
                "hyper_parameters": {},
                "raw": [],
            }

        this_data = parameter_data[name]
        summary = summary_iterator(str(file))
        for event in summary:
            for value in event.summary.value:
                if value.tag not in this_data:
                    this_data[value.tag] = {}

                this_data[value.tag][int(event.step)] = value.simple_value

                if value.tag == SESSION_START_INFO_TAG:
                    ssi = HParamsPluginData()
                    ssi.ParseFromString(value.metadata.plugin_data.content)
                    hparams = dict(ssi.session_start_info.hparams)
                    for k in hparams:
                        hparams[k] = hparams[k].ListFields()[0][1]
                    this_data["hyper_parameters"] = hparams

        break

    json_filename = f"{tensorboard_dir.parent.name}_data.json"
    if destination is None:
        file_path = tensorboard_dir.parent.joinpath(json_filename)
    else:
        if not destination.is_dir():
            destination.mkdir()
        file_path = destination.joinpath(json_filename)

    with open(file_path, "w") as file:
        file.write(json.dumps(parameter_data))

    return file_path


def list_failed(runtime_dir):
    runtime_dir = Path(runtime_dir)
    failed_list = []

    for filename in os.listdir(runtime_dir):
        with open(runtime_dir.joinpath(str(filename)), "r") as file:
            file_lines = file.readlines()
            return_code = int(file_lines[-1])

            if return_code == 0:
                continue

            command = file_lines[-3].split("::")[-1].strip()
            failed_list.append((filename, command))

    if len(failed_list) > 0:
        for filename, command in failed_list:
            # print(f"Failed {runtime_dir} :: {filename} :: {command}")
            print(f"!!! {runtime_dir}")


def data_from_models(models_dir: Path, destination: Path = None):
    if destination is None:
        destination = models_dir

    # if not destination.is_dir():
    #     destination.mkdir()
    if not models_dir.exists():
        print(f"!!! {models_dir}")
        return

    for run in next(os.walk(models_dir))[1]:
        data = {
            "str_weight_model": None,
            "str_nn_model": None,

            "parameters_json": None,
            "parameter_log": None,
            "loss_accuracy": None,

            "hyperparameters_weight_model": None,
            "hyperparameters_nn_model": None,
        }
        run_dir = models_dir.joinpath(str(run))
        run_files = sorted(os.listdir(run_dir), reverse=True)

        looking_for = list(data.keys())
        for i in run_files:
            if len(looking_for) == 0:
                break

            for j in looking_for:
                if j in str(i):
                    data[j] = i
                    looking_for.remove(j)
                    break

        if len(looking_for) > 0:
            print(f"Not Found {looking_for}")
            print(f"!!! {models_dir}")
            continue

        for key, value in data.items():
            data[key] = torch.load(run_dir.joinpath(value))

        if any([value is None for value in data.values()]):
            continue

        with open(destination.joinpath(f"{run}.json"), "w") as file:
            file.write(json.dumps(data))


def parse_data(input_path, output_path=None, multiple=False):
    input_path = Path(input_path)
    output_path = output_path or input_path.joinpath("json")
    output_path = Path(output_path)

    if not input_path.is_dir():
        raise Exception(f'"{input_path}" is not a directory or does not exists.')

    if multiple:
        all_input_path = [input_path.joinpath(x) for x in (next(os.walk(input_path))[1])]
    else:
        all_input_path = [input_path]

    for dir_path in all_input_path:
        print(f"Processing {dir_path}")

        if not dir_path.joinpath("runtime").exists():
            dir_path = dir_path.joinpath("_results")

        # list_failed(dir_path.joinpath("runtime"))
        # data_from_tensorboard(dir_path.joinpath("tensorboard"), output_path)
        data_from_models(dir_path.joinpath("models"), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--multiple", action='store_true')
    parser.set_defaults(multiple=False)
    all_arguments = parser.parse_known_args()
    kwargs = vars(all_arguments[0])
    parse_data(kwargs["source"], kwargs['output'], kwargs['multiple'])
