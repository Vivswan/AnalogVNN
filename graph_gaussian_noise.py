import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from nn.layers.ReducePrecision import ReducePrecision


def main():
    precision = 2 ** 2
    std = 1 / 100
    factor = 1
    poisson_lambda = 50

    xlim = [-1.1, 1.1]
    ylim = [-1.5, 1.5]

    num_line = torch.Tensor(np.linspace(-1, 1, 200))
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
    plt.title(f"Normal (std: {std}) * {factor}")

    plt.subplot(*sp, 5)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Possion (lamda: {poisson_lambda}) * {factor}")

    plt.subplot(*sp, 6)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.title(f"Normal + Poisson")

    for i in range(50):
        x = reduce_precision
        normal_noise = torch.normal(mean=x, std=std)
        poisson_noise = torch.sign(x) * (torch.poisson(torch.abs(x) * poisson_lambda) / poisson_lambda)
        normal_poisson_noise = torch.sign(normal_noise) * (
                torch.poisson(torch.abs(normal_noise) * poisson_lambda) / poisson_lambda)

        normal_noise = x + factor * (normal_noise - x)
        poisson_noise = x + factor * (poisson_noise - x)
        normal_poisson_noise = x + factor * (normal_poisson_noise - x)

        plt.subplot(*sp, 4)
        plt.scatter(plot_x, normal_noise.tolist(), color="#ff00000f", s=2)

        plt.subplot(*sp, 5)
        plt.scatter(plot_x, poisson_noise.tolist(), color="#ff00000f", s=2)

        plt.subplot(*sp, 6)
        plt.scatter(plot_x, normal_poisson_noise.tolist(), color="#ff00000f", s=2)

        plt.subplot(*sp, 7)
        plt.scatter(plot_x, rp_module(normal_noise).tolist(), color="#ff00000f", s=2)

        plt.subplot(*sp, 8)
        plt.scatter(plot_x, rp_module(poisson_noise).tolist(), color="#ff00000f", s=2)

        plt.subplot(*sp, 9)
        plt.scatter(plot_x, rp_module(normal_poisson_noise).tolist(), color="#ff00000f", s=2)

    fig.tight_layout()
    plt.show()
    fig.savefig('image.svg', dpi=fig.dpi)
    print()


if __name__ == '__main__':
    main()
