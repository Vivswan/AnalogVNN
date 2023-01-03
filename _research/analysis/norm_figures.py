import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from analogvnn.nn.normalize.Clamp import *
from analogvnn.nn.normalize.LPNorm import L1NormM, L2NormM, L1NormWM, L2NormWM


def main():
    n_pics = 4

    trainset = torchvision.datasets.CIFAR10(root='C:/_data/test/dataset', train=True, download=True,
                                            transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=n_pics ** 2, shuffle=True, num_workers=2)

    images, labels = iter(trainloader).next()

    norm_l1_norm_images = L1NormM()(images)
    norm_l2_norm_images = L2NormM()(images)
    norm_l1_norm_w_images = L1NormWM()(images)
    norm_l2_norm_w_images = L2NormWM()(images)

    # norm_l1_norm_images *= torch.mean(images) / torch.mean(norm_l1_norm_images)
    # norm_l2_norm_images *= torch.mean(images) / torch.mean(norm_l2_norm_images)
    # norm_l1_norm_w_images *= torch.mean(images) / torch.mean(norm_l1_norm_w_images)
    # norm_l2_norm_w_images *= torch.mean(images) / torch.mean(norm_l2_norm_w_images)

    # norm_l1_norm_images /= torch.max(norm_l1_norm_images)
    # norm_l2_norm_images /= torch.max(norm_l2_norm_images)
    # norm_l1_norm_w_images /= torch.max(norm_l1_norm_w_images)
    # norm_l2_norm_w_images /= torch.max(norm_l2_norm_w_images)

    # norm_l1_norm_images = (norm_l1_norm_images - images)
    # norm_l2_norm_images = (norm_l2_norm_images - images)
    # norm_l1_norm_w_images = (norm_l1_norm_w_images - images)
    # norm_l2_norm_w_images = (norm_l2_norm_w_images - images)

    images = torchvision.utils.make_grid(images, nrow=n_pics, padding=4)
    norm_l1_norm_images = torchvision.utils.make_grid(norm_l1_norm_images, nrow=n_pics, padding=4)
    norm_l2_norm_images = torchvision.utils.make_grid(norm_l2_norm_images, nrow=n_pics, padding=4)
    norm_l1_norm_w_images = torchvision.utils.make_grid(norm_l1_norm_w_images, nrow=n_pics, padding=4)
    norm_l2_norm_w_images = torchvision.utils.make_grid(norm_l2_norm_w_images, nrow=n_pics, padding=4)

    npimg = torch.cat((
        images,
        norm_l1_norm_images,
        norm_l2_norm_images,
        norm_l1_norm_w_images,
        norm_l2_norm_w_images,
    ), dim=2).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # imshow(to_image(images))


def create_text():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)

    fig.set_dpi(200)

    plt.title("Stochastic Reduce Precision Layer")
    plt.xlabel("Input")
    plt.ylabel("Output")
    fig.savefig(f'C:/_data/image.svg', dpi=200, transparent=True)


if __name__ == '__main__':
    # main()
    create_text()
