import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from parallel_main import run_combination_1, run_command


def run_combination_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    kwargs = vars(parser.parse_known_args()[0])

    print(f"data_folder: {kwargs['data_folder']}")

    data_folder = Path(kwargs['data_folder'])

    if not data_folder.exists():
        os.mkdir(data_folder)

    runtime = data_folder.joinpath("runtime")
    if not runtime.exists():
        os.mkdir(runtime)

    command_list = run_combination_1(kwargs['data_folder'], False)[:2]
    TOTAL_COUNT = len(command_list)
    print(f"number of combinations: {TOTAL_COUNT}")
    print()
    print()

    command_list = [(kwargs['data_folder'], x) for x in command_list]
    print(command_list)
    pool = Pool(2)
    pool.map(run_command, command_list)


if __name__ == '__main__':
    run_combination_main()
