import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from nn.layers.ReducePrecision import ReducePrecision


def main():
    torch.device('cpu')
    precision = 2 ** 2
    normal_reference_std = 1
    normal_reference_leakage = 1 - torch.erf(torch.tensor(normal_reference_std / math.sqrt(2)))
    normal_leakage = 0.1
    normal_std = 1 / (2 * precision * torch.erfinv(torch.tensor(1 - normal_leakage)) * math.sqrt(2))
    normal_snr = 1 / (normal_reference_std * 2 * normal_std)
    # factor = 1
    poisson_lambda = 50

    xlim = [-1.01, 1.01]
    ylim = [-1.5, 1.5]

    num_line = torch.Tensor(np.linspace(-1, 1, 1000))
    rp_module = ReducePrecision(precision=precision)
    reduce_precision = rp_module(num_line)

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
    plt.title(f"Normal (std: {normal_std:.4f}, snr: {normal_snr:.4})")

    plt.subplot(*sp, 5)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Possion (lamda: {poisson_lambda})")

    plt.subplot(*sp, 6)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Normal + Poisson")

    leakage_normal_noise = []
    leakage_poisson_noise = []
    leakage_normal_poisson_noise = []
    for i in range(100):
        x = reduce_precision
        x_size = sum(x.size())

        normal_noise = torch.normal(mean=x, std=normal_std)
        poisson_noise = torch.sign(x) * (torch.poisson(torch.abs(x) * poisson_lambda) / poisson_lambda)
        normal_poisson_noise = torch.sign(normal_noise) * (
                torch.poisson(torch.abs(normal_noise) * poisson_lambda) / poisson_lambda)

        rp_normal_noise = rp_module(normal_noise)
        rp_poisson_noise = rp_module(poisson_noise)
        rp_normal_poisson_noise = rp_module(normal_poisson_noise)

        leakage_normal_noise.append(torch.sum(torch.abs(x - rp_normal_noise)) * precision / x_size)
        leakage_poisson_noise.append(torch.sum(torch.abs(x - rp_poisson_noise)) * precision / x_size)
        leakage_normal_poisson_noise.append(torch.sum(torch.abs(x - rp_normal_poisson_noise)) * precision / x_size)

        if i < 5:
            plt.subplot(*sp, 4)
            plt.scatter(plot_x, normal_noise.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 5)
            plt.scatter(plot_x, poisson_noise.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 6)
            plt.scatter(plot_x, normal_poisson_noise.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 7)
            plt.scatter(plot_x, rp_normal_noise.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 8)
            plt.scatter(plot_x, rp_poisson_noise.tolist(), color="#ff00000f", s=2)

            plt.subplot(*sp, 9)
            plt.scatter(plot_x, rp_normal_poisson_noise.tolist(), color="#ff00000f", s=2)

    leakage_normal_noise = sum(leakage_normal_noise) / len(leakage_normal_noise)
    leakage_poisson_noise = sum(leakage_poisson_noise) / len(leakage_poisson_noise)
    leakage_normal_poisson_noise = sum(leakage_normal_poisson_noise) / len(leakage_normal_poisson_noise)

    print(f"leakage_normal_noise: {leakage_normal_noise * 100:.2f}%")
    print(f"leakage_poisson_noise: {leakage_poisson_noise * 100:.2f}%")
    print(f"leakage_normal_poisson_noise: {leakage_normal_poisson_noise * 100:.2f}%")

    fig.tight_layout()
    plt.show()
    fig.savefig('image.svg', dpi=fig.dpi)
    print()


if __name__ == '__main__':
    main()
