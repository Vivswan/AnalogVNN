import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path

from parallel_analogvnn1 import run_command, prepare_data_folder


def run_failed_combinations():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--failed_file", type=str, default=True)
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

    print(f"num_process: {kwargs['num_process']}")
    print(f"data_folder: {kwargs['data_folder']}")
    print(f"failed_file: {kwargs['failed_file']}")
    prepare_data_folder(kwargs['data_folder'])

    command_list = []
    with open(Path(kwargs["failed_file"]), "r") as file:
        for line in file.readlines():
            command_list.append(line.split("::")[-1])

    print(f"number of runs: {len(command_list)}")
    print()
    print()

    command_list = [(kwargs['data_folder'], x) for x in command_list]
    # print(command_list)

    if kwargs["single_run"]:
        command_list = command_list[:kwargs["num_process"]]

    with ThreadPool(kwargs["num_process"]) as pool:
        pool.map(run_command, command_list)


if __name__ == '__main__':
    run_failed_combinations()
