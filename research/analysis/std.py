import numpy as np
import torchvision

from research.dataloaders.load_vision_dataset import load_vision_dataset


def main():
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=torchvision.datasets.CIFAR10,
        path="C:/_data/test",
        batch_size=1,
        grayscale=False
    )

    sums_x = np.zeros(input_shape)
    sums_x2 = np.zeros(input_shape)
    for (data, target) in train_loader:
        data = np.array(data)
        sums_x += data
        sums_x2 += np.power(data, 2)
    for (data, target) in test_loader:
        data = np.array(data)
        sums_x += data
        sums_x2 += np.power(data, 2)

    n = np.shape(train_loader.dataset.data)[0] + np.shape(train_loader.dataset.data)[0]
    mean = sums_x / n
    mean = np.mean(mean)
    std = np.sqrt(np.sum(sums_x2 / (np.prod(input_shape) * n)) - (mean * mean))

    print(mean)
    print(std)


if __name__ == '__main__':
    main()
