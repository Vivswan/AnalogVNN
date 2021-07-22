import numpy as np
import torch
from matplotlib import pyplot as plt

from nn.layers.gaussian_noise_layer import GaussianNoise, TensorFunctions
from nn.layers.reduce_precision_layer import ReducePrecision

if __name__ == '__main__':
    precision = 8
    std = 0.1

    input = torch.Tensor(np.linspace(-1, 1, 100))
    x = input.tolist()

    rp_layer = ReducePrecision(precision=precision)
    rp_layer.eval()
    reduce_precision = rp_layer(input)
    plt.scatter(x, reduce_precision.tolist(), label="reduce_precision", color="#ff0000ff", s=2)
    for i in range(10):
        gaussian_noise = GaussianNoise(std=TensorFunctions.constant(std))(reduce_precision)
        plt.scatter(x, gaussian_noise.tolist(), color="#ff000055", s=2)

    plt.gca().set_xlim([-1.1, 1.1])
    plt.gca().set_ylim([-1.1, 1.1])

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(f'precision: {precision}, std: {std}')
    # plt.legend()

    plt.show()
    print()
