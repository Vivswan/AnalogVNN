import csv
import json
import math
from collections import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt

from nn.layers.GaussianNoise import GaussianNoise
from nn.layers.ReducePrecision import ReducePrecision
from nn.utils.is_using_cuda import get_device, set_device


def calculate_leakage(x, rp_normal_noise):
    return torch.sum(torch.sign(torch.abs(x - rp_normal_noise))) / sum(x.size())


def calculate_weighted_leakage(precision, x, rp_normal_noise):
    return torch.sum(torch.abs(x - rp_normal_noise)) * precision / sum(x.size())


def calculate_snr(signal, noise_signal):
    t2 = torch.tensor(2, device=get_device())
    s = torch.sum(torch.pow(signal, t2))
    n = torch.sum(torch.pow(signal - noise_signal, t2))
    return s / n


@dataclass
class CalculateResponds:
    signal: torch.Tensor
    rp: torch.Tensor
    noise_rp: torch.Tensor
    rp_noise_rp: torch.Tensor

    leakage: torch.Tensor
    weighted_leakage: torch.Tensor
    analog_snr: torch.Tensor
    digital_snr: torch.Tensor

    func: Callable


class CalculateRespondsAverages:
    def __init__(self):
        self.leakage: List[torch.Tensor] = []
        self.weighted_leakage: List[torch.Tensor] = []
        self.analog_snr: List[torch.Tensor] = []
        self.digital_snr: List[torch.Tensor] = []
        self._values = None

    def append_response(self, response: CalculateResponds):
        self.leakage.append(response.leakage)
        self.weighted_leakage.append(response.weighted_leakage)
        self.analog_snr.append(response.analog_snr)
        self.digital_snr.append(response.digital_snr)
        self._values = None

    @property
    def values(self):
        if self._values is None:
            self._values = (
                sum(self.leakage) / len(self.leakage),
                sum(self.weighted_leakage) / len(self.weighted_leakage),
                sum(self.analog_snr) / len(self.analog_snr),
                sum(self.digital_snr) / len(self.digital_snr)
            )
        return self._values

    def __repr__(self):
        return "leakage: {0:.4f}, weighted_leakage: {1:.4f}, analog_snr: {2:.4f}, digital_snr: {3:.4f}".format(
            *self.values)


def calculate(signal, precision, noise_fn) -> CalculateResponds:
    rp_signal = ReducePrecision(precision=precision)(signal)
    noise_rp_signal = noise_fn(rp_signal)
    rp_noise_rp_signal = ReducePrecision(precision=precision)(noise_rp_signal)

    return CalculateResponds(
        signal=signal,
        rp=rp_signal,
        noise_rp=noise_rp_signal,
        rp_noise_rp=rp_noise_rp_signal,

        leakage=calculate_leakage(rp_signal, rp_noise_rp_signal),
        weighted_leakage=calculate_weighted_leakage(precision, rp_signal, rp_noise_rp_signal),
        analog_snr=calculate_snr(signal, noise_rp_signal),
        digital_snr=calculate_snr(signal, rp_noise_rp_signal),
        func=noise_fn,
    )


def create_normal_snr():
    folder_path = Path("C:/_data")
    set_device('cpu')
    leakages = np.linspace(0, 1, 101)
    precisions = np.array(range(1, 2 ** 8))

    x_values = torch.Tensor(np.linspace(-1, 1, 1000))

    data = [
        ["precision", "leakage", "calculated_leakages", "weighted_leakage", "analog_snr", "digital_snr"],
    ]
    result_str = ""
    for i, precision in enumerate(precisions):
        for j, leakage in enumerate(leakages):
            average_response = CalculateRespondsAverages()
            for k in range(1000):
                average_response.append_response(calculate(
                    x_values, float(precision),
                    lambda x: GaussianNoise(leakage=leakage, precision=precision)(x)
                ))
            data.append([precision, leakage] + [float(x) for x in average_response.values])
            print_str = f"{i}, {j}: {abs(average_response.values[0] - leakage):.4f} == 0, precision: {precision}, leakage: {leakage:.4f}, {average_response}"
            print(print_str)
            result_str += print_str + "\n"

    with open(folder_path.joinpath("normal_noise.txt"), "w") as file:
        file.write(result_str)
    with open(folder_path.joinpath("normal_noise.json"), "w") as file:
        file.write(json.dumps(data))
    with open(folder_path.joinpath("normal_noise.csv"), 'w', newline='') as file:
        csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL).writerows(data)

    data_shape = (precisions.size, leakages.size)
    calculated_leakages = np.zeros(data_shape)
    weighted_leakage = np.zeros(data_shape)
    analog_snr = np.zeros(data_shape)
    digital_snr = np.zeros(data_shape)

    for i, v in enumerate(data[1:]):
        precision_index = int(np.where(precisions == v[0])[0])
        leakage_index = int(np.where(leakages == v[1])[0])
        calculated_leakages[precision_index][leakage_index] = v[2]
        weighted_leakage[precision_index][leakage_index] = v[3]
        analog_snr[precision_index][leakage_index] = v[4]
        digital_snr[precision_index][leakage_index] = v[5]

    scipy.io.savemat(folder_path.joinpath("normal_noise.mat"), {
        "leakages": leakages,
        "precisions": precisions,
        "calculated_leakages": calculated_leakages,
        "weighted_leakage": weighted_leakage,
        "analog_snr": analog_snr,
        "digital_snr": digital_snr,
    })


def main():
    set_device('cpu')
    precision = 2 ** 2
    normal_leakage = 0.1
    poisson_lambda = 50

    xlim = [-1.01, 1.01]
    ylim = [-1.5, 1.5]

    num_line = torch.Tensor(np.linspace(-1, 1, 1000))
    reduce_precision = ReducePrecision(precision=precision)(num_line)

    sp = [3, 3]
    fig, axes = plt.subplots(nrows=sp[0], ncols=sp[1])
    fig.set_size_inches(14, 11)
    fig.set_dpi(200)

    plot_x = num_line.tolist()
    plt.subplot(*sp, 2)
    plt.scatter(plot_x, reduce_precision.tolist(), label="reduce_precision", color="#ff0000", s=2)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Reduce Precision (p: {precision} = {math.log2(precision):0.3f} bits)")

    plt.subplot(*sp, 4)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Normal")

    plt.subplot(*sp, 5)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Possion (lamda: {poisson_lambda})")

    plt.subplot(*sp, 6)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Normal + Poisson")

    normal_avg = CalculateRespondsAverages()
    poisson_avg = CalculateRespondsAverages()
    normal_poisson_avg = CalculateRespondsAverages()

    for i in range(1000):
        normal_noise = calculate(
            num_line, precision,
            lambda x: GaussianNoise(leakage=normal_leakage, precision=precision)(x)
        )
        poisson_noise = calculate(
            num_line, precision,
            lambda x: torch.sign(x) * (torch.poisson(torch.abs(x) * poisson_lambda) / poisson_lambda)
        )
        normal_poisson_noise = calculate(
            num_line, precision,
            lambda x: normal_noise.func(poisson_noise.func(x))
        )

        normal_avg.append_response(normal_noise)
        poisson_avg.append_response(poisson_noise)
        normal_poisson_avg.append_response(normal_poisson_noise)

        if i < 5:
            plt.subplot(*sp, 4)
            plt.scatter(plot_x, normal_noise.noise_rp.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 5)
            plt.scatter(plot_x, poisson_noise.noise_rp.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 6)
            plt.scatter(plot_x, normal_poisson_noise.noise_rp.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 7)
            plt.scatter(plot_x, normal_noise.rp_noise_rp.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 8)
            plt.scatter(plot_x, poisson_noise.rp_noise_rp.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 9)
            plt.scatter(plot_x, normal_poisson_noise.rp_noise_rp.tolist(), color="#ff00000f", s=2)

    print(f"normal_avg: {normal_avg}")
    print(f"poisson_avg: {poisson_avg}")
    print(f"normal_poisson_avg: {normal_poisson_avg}")

    fig.tight_layout()
    plt.show()
    # fig.savefig('image.svg', dpi=fig.dpi)
    print()


if __name__ == '__main__':
    main()
