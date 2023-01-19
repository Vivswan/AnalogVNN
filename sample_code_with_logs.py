import json
from pathlib import Path

import numpy as np
import torch.backends.cudnn
import torchvision
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from analogvnn.nn.Linear import Linear
from analogvnn.nn.activation.Gaussian import GeLU
from analogvnn.nn.module.FullSequential import FullSequential
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Clamp import Clamp
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.parameter.PseudoParameter import PseudoParameter
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda
from analogvnn.utils.render_autograd_graph import save_autograd_graph_from_module


def load_vision_dataset(dataset, path, batch_size, is_cuda=False, grayscale=True):
    """Loads a vision dataset with optional grayscale conversion and CUDA support.

    Args:
        dataset (Type[torchvision.datasetsVisionDataset]): the dataset class to use (e.g. torchvision.datasets.MNIST)
        path (str): the path to the dataset files
        batch_size (int): the batch size to use for the data loader
        is_cuda (bool): a flag indicating whether to use CUDA support (defaults to False)
        grayscale (bool): a flag indicating whether to convert the images to grayscale (defaults to True)

    Returns:
        A tuple containing the train and test data loaders, the input shape, and a tuple of class labels.
    """

    dataset_kwargs = {
        'batch_size': batch_size,
        'shuffle': True
    }

    if is_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
        }
        dataset_kwargs.update(cuda_kwargs)

    if grayscale:
        use_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    else:
        use_transform = transforms.Compose([transforms.ToTensor()])

    train_set = dataset(root=path, train=True, download=True, transform=use_transform)
    test_set = dataset(root=path, train=False, download=True, transform=use_transform)
    train_loader = DataLoader(train_set, **dataset_kwargs)
    test_loader = DataLoader(test_set, **dataset_kwargs)

    zeroth_element = next(iter(test_loader))[0]
    input_shape = list(zeroth_element.shape)

    return train_loader, test_loader, input_shape, tuple(train_set.classes)


def cross_entropy_accuracy(output, target) -> float:
    """Cross Entropy accuracy function.

    Args:
        output (torch.Tensor): output of the model from passing inputs
        target (torch.Tensor): correct labels for the inputs

    Returns:
        float: accuracy from 0 to 1
    """

    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return correct / len(output)


class LinearModel(FullSequential):
    def __init__(self, activation_class, norm_class, precision_class, precision, noise_class, leakage):
        """Linear Model with 3 dense neural network layers.

        Args:
            activation_class: Activation Class
            norm_class: Normalization Class
            precision_class: Precision Class (ReducePrecision or StochasticReducePrecision)
            precision (int): precision of the weights and biases
            noise_class: Noise Class
            leakage (float): leakage is the probability that a reduced precision digital value (e.g., “1011”) will
            acquire a different digital value (e.g., “1010” or “1100”) after passing through the noise layer
            (i.e., the probability that the digital values transmitted and detected are different after passing through
            the analog channel).
        """

        super(LinearModel, self).__init__()

        self.activation_class = activation_class
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers = []
        self.all_layers.append(nn.Flatten(start_dim=1))
        self.add_layer(Linear(in_features=28 * 28, out_features=256))
        self.add_layer(Linear(in_features=256, out_features=128))
        self.add_layer(Linear(in_features=128, out_features=10))

        self.add_sequence(*self.all_layers)

    def add_layer(self, layer):
        """To add the analog layer.

        Args:
            layer (BaseLayer): digital layer module
        """

        self.all_layers.append(self.norm_class())
        self.all_layers.append(self.precision_class(precision=self.precision))
        self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
        self.all_layers.append(layer)
        self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
        self.all_layers.append(self.norm_class())
        self.all_layers.append(self.precision_class(precision=self.precision))
        self.all_layers.append(self.activation_class())
        self.activation_class.initialise_(layer.weight)


class WeightModel(FullSequential):
    def __init__(self, norm_class, precision_class, precision, noise_class, leakage):
        """Weight Model.

        Args:
            norm_class: Normalization Class
            precision_class: Precision Class (ReducePrecision or StochasticReducePrecision)
            precision (int): precision of the weights and biases
            noise_class: Noise Class
            leakage (float): leakage is the probability that a reduced precision digital value (e.g., “1011”) will
            acquire a different digital value (e.g., “1010” or “1100”) after passing through the noise layer
            (i.e., the probability that the digital values transmitted and detected are different after passing through
            the analog channel).
        """

        super(WeightModel, self).__init__()
        self.all_layers = []

        self.all_layers.append(norm_class())
        self.all_layers.append(precision_class(precision=precision))
        self.all_layers.append(noise_class(leakage=leakage, precision=precision))

        self.eval()
        self.add_sequence(*self.all_layers)


def run_linear3_model():
    """The main function to train photonics image classifier with 3 linear/dense nn for MNIST dataset."""

    is_cpu_cuda.use_cuda_if_available()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    data_path = Path('_data')
    if not data_path.exists():
        data_path.mkdir()

    device, is_cuda = is_cpu_cuda.is_using_cuda
    print(f'Device: {device}')
    print()

    # Loading Data
    print('Loading Data...')
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=torchvision.datasets.MNIST,
        path=str(data_path),
        batch_size=128,
        is_cuda=is_cuda
    )

    # Creating Models
    print('Creating Models...')
    nn_model = LinearModel(
        activation_class=GeLU,
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.5
    )
    weight_model = WeightModel(
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.5
    )

    # Parametrizing Parameters of the Models
    PseudoParameter.parametrize_module(nn_model, transformation=weight_model)

    # Setting Model Parameters
    nn_model.loss_function = nn.CrossEntropyLoss()
    nn_model.accuracy_function = cross_entropy_accuracy
    nn_model.optimizer = optim.Adam(params=nn_model.parameters())

    # Compile Model
    nn_model.compile(device=device)
    weight_model.compile(device=device)

    # Creating Tensorboard for Visualization and Saving the Model
    nn_model.create_tensorboard(str(data_path.joinpath('tensorboard')))

    print('Saving Summary and Graphs...')
    _, nn_model_summary = nn_model.tensorboard.add_summary(train_loader=train_loader)
    _, weight_model_summary = nn_model.tensorboard.add_summary(train_loader=train_loader, model=weight_model)
    save_autograd_graph_from_module(data_path.joinpath('nn_model'), nn_model, next(iter(train_loader))[0])
    save_autograd_graph_from_module(data_path.joinpath('weight_model'), weight_model, torch.ones((1, 1)))
    save_autograd_graph_from_module(
        data_path.joinpath('nn_model_autograd'),
        nn_model,
        next(iter(train_loader))[0],
        from_forward=True
    )
    save_autograd_graph_from_module(
        data_path.joinpath('weight_model_autograd'),
        weight_model,
        torch.ones((1, 1)),
        from_forward=True
    )
    nn_model.tensorboard.add_graph(train_loader)
    nn_model.tensorboard.add_graph(train_loader, model=weight_model)

    print('Creating Log File...')
    with open(data_path.joinpath('logfile.log'), 'a+', encoding='utf-8') as file:
        file.write(str(nn_model.optimizer) + '\n\n')

        file.write(str(nn_model) + '\n\n')
        file.write(str(weight_model) + '\n\n')
        file.write(f'{nn_model_summary}\n\n')
        file.write(f'{weight_model_summary}\n\n')
    loss_accuracy = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }

    # Training
    print('Starting Training...')
    for epoch in range(10):
        train_loss, train_accuracy = nn_model.train_on(train_loader, epoch=epoch)
        test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch)

        loss_accuracy['train_loss'].append(train_loss)
        loss_accuracy['train_accuracy'].append(train_accuracy)
        loss_accuracy['test_loss'].append(test_loss)
        loss_accuracy['test_accuracy'].append(test_accuracy)

        str_epoch = str(epoch + 1).zfill(1)
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        with open(data_path.joinpath('logfile.log'), 'a+') as file:
            file.write(print_str)
    print('Run Completed Successfully...')

    metric_dict = {
        'train_loss': loss_accuracy['train_loss'][-1],
        'train_accuracy': loss_accuracy['train_accuracy'][-1],
        'test_loss': loss_accuracy['test_loss'][-1],
        'test_accuracy': loss_accuracy['test_accuracy'][-1],
        'min_train_loss': np.min(loss_accuracy['train_loss']),
        'max_train_accuracy': np.max(loss_accuracy['train_accuracy']),
        'min_test_loss': np.min(loss_accuracy['test_loss']),
        'max_test_accuracy': np.max(loss_accuracy['test_accuracy']),
    }
    nn_model.tensorboard.tensorboard.add_hparams(
        metric_dict=metric_dict,
        hparam_dict={
            'precision': 2 ** 4,
            'leakage': 0.5,
            'noise': 'GaussianNoise',
            'activation': 'GeLU',
            'precision_class': 'ReducePrecision',
            'norm_class': 'Clamp',
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'accuracy_function': 'cross_entropy_accuracy',
            'batch_size': 128,
            'epochs': 1,
        }
    )

    with open(data_path.joinpath('logfile.log'), 'a+') as file:
        file.write(json.dumps(metric_dict, sort_keys=True) + '\n\n')
        file.write('Run Completed Successfully...')
    nn_model.tensorboard.close()
    print()


if __name__ == '__main__':
    run_linear3_model()
