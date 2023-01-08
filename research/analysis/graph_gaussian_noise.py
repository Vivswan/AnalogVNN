import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable

import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.io import savemat

from analogvnn.fn.dirac_delta import dirac_delta
from analogvnn.nn.module.Sequential import Sequential
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.noise.LaplacianNoise import LaplacianNoise
from analogvnn.nn.noise.PoissonNoise import PoissonNoise
from analogvnn.nn.noise.UniformNoise import UniformNoise
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda


def calculate_leakage(x, rp_normal_noise):
    return torch.sum(torch.sign(torch.abs(x - rp_normal_noise))) / np.prod(x.size())


def calculate_weighted_leakage(precision, x, rp_normal_noise):
    return torch.sum(torch.abs(x - rp_normal_noise)) * precision / np.prod(x.size())


def calculate_snr(signal, noise_signal):
    t2 = torch.tensor(2, device=signal.device)
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
    digital_analog_snr: torch.Tensor
    analog_analog_snr: torch.Tensor
    digital_digital_snr: torch.Tensor

    func: Callable


class CalculateRespondsAverages:
    def __init__(self):
        self.leakage: List[torch.Tensor] = []
        self.weighted_leakage: List[torch.Tensor] = []
        self.digital_analog_snr: List[torch.Tensor] = []
        self.analog_analog_snr: List[torch.Tensor] = []
        self.digital_digital_snr: List[torch.Tensor] = []
        self._values = None

    def append_response(self, response: CalculateResponds):
        self.leakage.append(response.leakage)
        self.weighted_leakage.append(response.weighted_leakage)
        self.digital_analog_snr.append(response.digital_analog_snr)
        self.analog_analog_snr.append(response.analog_analog_snr)
        self.digital_digital_snr.append(response.digital_digital_snr)
        self._values = None

    @property
    def values(self):
        if self._values is None:
            self._values = (
                sum(self.leakage) / len(self.leakage),
                sum(self.weighted_leakage) / len(self.weighted_leakage),
                sum(self.digital_analog_snr) / len(self.digital_analog_snr),
                sum(self.analog_analog_snr) / len(self.analog_analog_snr),
                sum(self.digital_digital_snr) / len(self.digital_digital_snr)
            )
        return self._values

    def __repr__(self):
        return "leakage: {0:.4f}" \
               ", weighted_leakage: {1:.4f}" \
               ", digital_analog_snr: {2:.4f}" \
               ", analog_analog_snr: {3:.4f}" \
               ", digital_digital_snr: {4:.4f}" \
               "".format(*self.values)


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
        digital_analog_snr=calculate_snr(signal, noise_rp_signal),
        analog_analog_snr=calculate_snr(rp_signal, noise_rp_signal),
        digital_digital_snr=calculate_snr(signal, rp_noise_rp_signal),
        func=noise_fn,
    )


def create_normal_snr():
    folder_path = Path("C:/_data")
    is_cpu_cuda.set_device('cpu')
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


def main(std=0.1, precision=3):
    is_cpu_cuda.set_device('cpu')

    xlim = [-1.01, 1.01]
    ylim = [-1.5, 1.5]

    num_line = torch.Tensor(np.linspace(-1, 1, 500))
    reduce_precision = ReducePrecision(precision=precision)(num_line)

    sp = [3, 3]
    fig, axes = plt.subplots(nrows=sp[0], ncols=sp[1])
    # fig.set_size_inches(4.0, 4.0)
    fig.set_size_inches(6, 6)
    fig.set_dpi(200)

    plot_x = num_line.tolist()
    plt.subplot(*sp, 1)
    plt.plot(plot_x, plot_x, color="#ff0000")
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.subplot(*sp, 3)
    plt.plot(plot_x, (np.array(plot_x) * 1.5).tolist(), color="#ff0000")
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    plt.subplot(*sp, 2)
    plt.plot(plot_x, reduce_precision.tolist(), label="reduce_precision", color="#ff0000")
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
    plt.title(f"Possion (scale: {std})")

    plt.subplot(*sp, 6)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Normal + Poisson")

    uniform_avg = CalculateRespondsAverages()
    normal_avg = CalculateRespondsAverages()
    poisson_avg = CalculateRespondsAverages()
    poisson_ones_avg = CalculateRespondsAverages()
    laplace_avg = CalculateRespondsAverages()
    normal_poisson_avg = CalculateRespondsAverages()

    un = UniformNoise(leakage=std, precision=precision)
    gn = GaussianNoise(leakage=std, precision=precision)
    pn = PoissonNoise(max_leakage=0.1, precision=precision)
    ln = LaplacianNoise(scale=1 / std, precision=precision)
    for i in range(1000):
        uniform_noise = calculate(num_line, precision, un)
        normal_noise = calculate(num_line, precision, gn)
        poisson_noise = calculate(num_line, precision, pn)
        poisson_noise_ones = calculate(torch.ones_like(num_line), precision, pn)
        laplace_noise = calculate(num_line, precision, ln)
        normal_poisson_noise = calculate(
            num_line, precision,
            lambda x: normal_noise.func(poisson_noise.func(x))
        )

        uniform_avg.append_response(uniform_noise)
        normal_avg.append_response(normal_noise)
        poisson_avg.append_response(poisson_noise)
        poisson_ones_avg.append_response(poisson_noise_ones)
        laplace_avg.append_response(laplace_noise)
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

    plt.subplot(*sp, 7)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.subplot(*sp, 8)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.subplot(*sp, 9)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    print(f"std: {std:0.4f}, precision: {precision}")
    print(f"uniform_expected: {un.leakage:0.4f}, {uniform_avg}")
    print(f"normal_expected: {gn.leakage:0.4f}, {normal_avg}")
    print(f"poisson_expected: {pn.leakage:0.4f}, {poisson_avg}")
    print(f"poisson_ones_expected: {pn.max_leakage:0.4f}, {poisson_ones_avg}")
    print(f"laplace_expected: {ln.leakage:0.4f}, {laplace_avg}")
    print(f"normal_poisson_avg: {normal_poisson_avg}")
    #
    fig.tight_layout()
    plt.show()
    fig.savefig('C:/_data/image.svg', dpi=fig.dpi)
    print()


def plot_and_integrate(std=0.2, precision=2):
    reduce_precision = ReducePrecision(precision=precision)
    gaussian_noise = GaussianNoise(std=std, precision=precision)
    poisson_noise = PoissonNoise(scale=1 / std, precision=precision)

    def normal_int_fn(p, q):
        rp_q = reduce_precision(q)
        return np.sign(dirac_delta(reduce_precision(p) - rp_q)) * gaussian_noise.pdf(p, loc=rp_q)

    def poisson_int_fn(p, q):
        rp_q = reduce_precision(q)
        return np.sign(dirac_delta(reduce_precision(p) - rp_q)) * poisson_noise.pdf(p, rate=abs(rp_q))

    def p_threshold(q, a=-np.inf, b=np.inf):
        return [
            reduce_precision(q) - reduce_precision.precision_width / 2,
            reduce_precision(q) + reduce_precision.precision_width / 2
        ]
        # return [
        #     np.maximum(float(reduce_precision(q) - reduce_precision.step_width / 2), a),
        #     np.minimum(float(reduce_precision(q) + reduce_precision.step_width / 2), b)
        # ]

    k = int(reduce_precision.precision.data) + 1
    num_line_domain = [-1.01, 1.01]
    num_line = np.linspace(*num_line_domain, 250, dtype=float)

    plt.figure()
    fig, axes = plt.subplots(k + 1, 1)
    fig.set_size_inches(5, 7.5)
    fig.set_dpi(500)
    plt.subplot(k + 1, 1, 1)

    # n_int_normal_correct = integrate.nquad(normal_int_fn, [p_threshold, num_line_domain])[0] / np.sum(np.abs(num_line_domain))
    # n_int_poisson_correct = integrate.nquad(poisson_int_fn, [p_threshold, num_line_domain])[0] / np.sum(np.abs(num_line_domain))

    normal_correct = []
    poisson_correct = []
    for i in num_line:
        normal_correct.append(
            integrate.quad(gaussian_noise.pdf, *p_threshold(i, -1., 1.), args=(reduce_precision(i),))[0])
        poisson_correct.append(
            integrate.quad(poisson_noise.pdf, *p_threshold(i, -1., 1.), args=(reduce_precision(i),))[0])

    # plt.plot(num_line, abs(reduce_precision(num_line)), label="abs reduce_precision")
    plt.plot(num_line, normal_correct, label="normal_correct")
    plt.plot(num_line, poisson_correct, label="poisson_correct")

    plt.ylim(-0.01, 1.1)
    plt.legend()
    for i in range(k):
        plt.subplot(k + 1, 1, i + 2)
        plt.plot(num_line, normal_int_fn(num_line, i * reduce_precision.precision_width), label=f"normal_int {i}")
        plt.plot(num_line, poisson_int_fn(num_line, i * reduce_precision.precision_width), label=f"poisson_int {i}")
        plt.plot(num_line, normal_int_fn(num_line, -i * reduce_precision.precision_width), label=f"normal_int {-i}")
        plt.plot(num_line, poisson_int_fn(num_line, -i * reduce_precision.precision_width), label=f"poisson_int {-i}")
        plt.ylim(-0.01, 1.1)
        plt.legend()
    fig.tight_layout()
    plt.show()
    # fig.savefig('image.svg', dpi=fig.dpi)

    pdf_normal_correct = 0
    cdf_normal_correct = 0
    pdf_poisson_correct = 0
    cdf_poisson_correct = 0
    cdf_domain = np.linspace(-1, 1, 2 * precision + 1, dtype=float)
    print(precision)
    print(cdf_domain)
    print(reduce_precision(cdf_domain))
    for i in cdf_domain:
        min_i = i - reduce_precision.precision_width / 2
        max_i = i + reduce_precision.precision_width / 2
        if np.isclose(i, 1.):
            max_i = 1.
        if np.isclose(i, -1.):
            min_i = -1.

        # print(f"N {i:0.4f}: {float(gaussian_noise.cdf(max_i, mean=i) - gaussian_noise.cdf(min_i, mean=i)):0.5f}")
        pdf_normal_correct += integrate.quad(gaussian_noise.pdf, min_i, max_i, args=(reduce_precision(i),))[0]
        cdf_normal_correct += gaussian_noise.cdf(max_i, mean=reduce_precision(i))
        cdf_normal_correct -= gaussian_noise.cdf(min_i, mean=reduce_precision(i))

        pdf_poisson_correct += integrate.quad(poisson_noise.pdf, min_i, max_i, args=(reduce_precision(i),))[0]
        print(f"P {np.isclose(i, reduce_precision(i))}, {i:0.4f}"
              f" : [{max(abs(max_i), abs(min_i)):0.4f}, {min(abs(max_i), abs(min_i)):0.4f}]"
              f" : {float(poisson_noise.cdf(max(abs(max_i), abs(min_i)), rate=i)):0.5f}"
              f" - {float(poisson_noise.cdf(min(abs(max_i), abs(min_i)), rate=i)):0.5f}"
              f" = {float(poisson_noise.cdf(max(abs(max_i), abs(min_i)), rate=i) - poisson_noise.cdf(min(abs(max_i), abs(min_i)), rate=i)):0.5f}")

        if np.isclose(i, 0.):
            cdf_poisson_correct += poisson_noise.cdf(max(abs(max_i), abs(min_i)), rate=i)
        else:
            cdf_poisson_correct += poisson_noise.cdf(max(abs(max_i), abs(min_i)), rate=i)
            cdf_poisson_correct -= poisson_noise.cdf(min(abs(max_i), abs(min_i)), rate=i)

    pdf_normal_correct /= cdf_domain.size - 1
    cdf_normal_correct /= cdf_domain.size - 1
    pdf_poisson_correct /= cdf_domain.size - 1
    cdf_poisson_correct /= cdf_domain.size - 1

    print(
        f"normal_correct"
        f" - expected: {1 - gaussian_noise.leakage:0.6f}"
        f", got numerically: {np.mean(normal_correct):0.6f}"
        # f", got integrally: {n_int_normal_correct:0.6f}"
        f", got pdf: {pdf_normal_correct:0.6f}"
        f", got cdf: {cdf_normal_correct:0.6f}"
    )
    print(
        f"poisson_correct"
        f" - expected: {1 - poisson_noise.leakage:0.6f}"
        f", got numerically: {np.mean(poisson_correct):0.6f}"
        # f", got integrally: {n_int_poisson_correct:0.6f}"
        f", got pdf: {pdf_poisson_correct:0.6f}"
        f", got cdf: {cdf_poisson_correct:0.6f}"
    )
    print(
        f"normal_leakage"
        f" - expected: {gaussian_noise.leakage:0.6f}"
        f", got numerically: {1 - np.mean(normal_correct):0.6f}"
        f", got pdf: {1 - pdf_normal_correct:0.6f}"
        f", got cdf: {1 - cdf_normal_correct:0.6f}"
    )
    print(
        f"poisson_leakage"
        f" - expected: {poisson_noise.leakage:0.6f}"
        f", got numerically: {1 - np.mean(poisson_correct):0.6f}"
        f", got pdf: {1 - pdf_poisson_correct:0.6f}"
        f", got cdf: {1 - cdf_poisson_correct:0.6f}"
    )
    print(sorted(set(poisson_correct)))
    main(std=std, precision=precision)


def plot_poisson_leakage():
    max_bit = 6
    precision = np.linspace(2 ** 0, 2 ** max_bit, 2 ** max_bit)
    scale = np.linspace(0, 20, 210)
    leakage = np.zeros((precision.size, scale.size))
    for i, p in enumerate(precision):
        for j, s in enumerate(scale):
            leakage[i][j] = PoissonNoise.staticmethod_leakage(scale=s, precision=p)

    scale, precision = np.meshgrid(scale, precision)
    savemat("../_data/plot_poisson_leakage.mat", {
        "precision": precision,
        "scale": scale,
        "leakage": leakage,
    })
    plt.contourf(scale, precision, leakage, 100)
    plt.colorbar()
    plt.show()


def plot_normal_leakage():
    max_bit = 3
    precision = np.linspace(2 ** 0, 2 ** max_bit, 2 ** max_bit)
    std = np.linspace(0, 1, 500)
    leakage = np.zeros((precision.size, std.size))
    for i, p in enumerate(precision):
        for j, s in enumerate(std):
            leakage[i][j] = GaussianNoise.calc_leakage(std=s, precision=p)

    std, precision = np.meshgrid(std, precision)
    savemat("../_data/plot_normal_leakage.mat", {
        "precision": precision,
        "std": std,
        "leakage": leakage,
    })
    plt.contourf(std, precision, leakage, 100)
    plt.colorbar()
    plt.show()


@torch.no_grad()
def plot_layer_leakage():
    max_bit = 3
    precision = np.linspace(2 ** 0, 2 ** max_bit, 2 ** max_bit)
    std = np.linspace(0, 3, 30)
    scale = np.linspace(0, 5, 50)

    rp = ReducePrecision(precision=1)
    gn = GaussianNoise(std=1, precision=1)
    pn = PoissonNoise(scale=1, precision=1)

    layer = Sequential(
        rp,
        gn,
        pn,
        # Identity(),
        # pn,
        # gn,
        rp,
    )

    leakage = np.zeros((precision.size, std.size, scale.size))
    for i, p in enumerate(precision):
        for j, s in enumerate(std):
            print(f"{i}, {j}")
            for k, sc in enumerate(scale):
                # rp.precision.data = torch.tensor(p)
                gn.std.data = torch.tensor(s)
                pn.precision.data = torch.tensor(p)
                pn.scale.data = torch.tensor(sc)

                temp = torch.rand((2000, 2000))
                leakage[i][j][k] = float(calculate_leakage(rp(temp), layer(temp)))

    savemat("../_data/plot_layer_leakage.mat", {
        "precision": precision,
        "std": std,
        "scale": scale,
        "leakage": leakage,
    })
    # std, precision, scale = np.meshgrid(std, precision)
    #
    # plt.contourf(std, precision, leakage, 100)
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
    # plot_and_integrate()
    # plot_poisson_leakage()
    # plot_normal_leakage()
    # plot_layer_leakage()
