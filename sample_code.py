from typing import Type, List

import torch.backends.cudnn
import torchvision
from torch import optim
from torch.nn import Flatten

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.activations.Activation import Activation
from nn.activations.Gaussian import GeLU
from nn.layers.Linear import Linear
from nn.layers.functionals.BackwardWrapper import BackwardWrapper
from nn.layers.functionals.Normalize import *
from nn.layers.functionals.ReducePrecision import ReducePrecision
from nn.layers.functionals.StochasticReducePrecision import StochasticReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.modules.FullSequential import FullSequential
from nn.modules.Sequential import Sequential
from nn.optimizer.PseudoOptimizer import PseudoOptimizer
from nn.utils.is_cpu_cuda import is_cpu_cuda


def cross_entropy_loss_accuracy(output, target):
    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return correct / len(output)


class LinearModel(FullSequential):
    def __init__(
            self,
            activation_class: Type[Activation],
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(LinearModel, self).__init__()

        self.activation_class = activation_class
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers: List[nn.Module] = []
        self.all_layers.append(BackwardWrapper(Flatten(start_dim=1)))
        self.add_layer(Linear(in_features=28 * 28, out_features=256))
        self.add_layer(Linear(in_features=256, out_features=128))
        self.add_layer(Linear(in_features=128, out_features=10))

        self.add_sequence(*self.all_layers)

    def add_layer(self, layer):
        self.add_doa_layers()
        self.all_layers.append(layer)
        self.add_aod_layers()
        self.activation_class.initialise_(layer.weight)

    def add_doa_layers(self):
        if self.norm_class is not None:
            self.all_layers.append(self.norm_class())
        if self.precision_class is not None:
            self.all_layers.append(self.precision_class(precision=self.precision))
        if self.noise_class is not None:
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))

    def add_aod_layers(self):
        if self.noise_class is not None:
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
        if self.norm_class is not None:
            self.all_layers.append(self.norm_class())
        if self.precision_class is not None:
            self.all_layers.append(self.precision_class(precision=self.precision))

        if self.activation_class is not None:
            self.all_layers.append(self.activation_class())

    def set_optimizer(self, optimizer_cls, super_optimizer_cls=None, param_sanitizer=None, **optimiser_parameters):
        if super_optimizer_cls is not None:
            self.optimizer = super_optimizer_cls(
                optimizer_cls=optimizer_cls,
                params=self.parameters() if param_sanitizer is None else param_sanitizer(self.parameters()),
                **optimiser_parameters
            )
        else:
            self.optimizer = optimizer_cls(
                params=self.parameters() if param_sanitizer is None else param_sanitizer(self.parameters()),
                **optimiser_parameters
            )
        return self


class WeightModel(Sequential):
    def __init__(
            self,
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(WeightModel, self).__init__()
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers = []

        if norm_class is not None:
            self.all_layers.append(norm_class())
        if precision_class is not None:
            self.all_layers.append(precision_class(precision=precision))
        if noise_class is not None:
            self.all_layers.append(noise_class(leakage=leakage, precision=precision))

        self.eval()
        if len(self.all_layers) > 0:
            self.add_sequence(*self.all_layers)


def run_linear3_model():
    torch.backends.cudnn.benchmark = True
    device, is_cuda = is_cpu_cuda.is_using_cuda()
    print(f"Device: {device}")
    print()

    print(f"Loading Data...")
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=torchvision.datasets.MNIST,
        path="_data/",
        batch_size=128,
        is_cuda=is_cuda
    )

    print(f"Creating Models...")
    nn_model = LinearModel(
        activation_class=GeLU,
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.2
    )
    weight_model = WeightModel(
        norm_class=Clamp,
        precision_class=ReducePrecision,
        precision=2 ** 4,
        noise_class=GaussianNoise,
        leakage=0.2
    )

    nn_model.compile(device=device, layer_data=True)
    nn_model.loss_fn = nn.CrossEntropyLoss()
    nn_model.accuracy_fn = cross_entropy_loss_accuracy
    nn_model.to(device=device)
    weight_model.to(device=device)
    PseudoOptimizer.parameter_type.convert_model(nn_model, transform=weight_model)
    nn_model.set_optimizer(
        super_optimizer_cls=PseudoOptimizer,
        optimizer_cls=optim.Adam,
    )

    print(f"Starting Training...")
    for epoch in range(10):
        train_loss, train_accuracy = nn_model.train_on(train_loader, epoch=epoch)
        test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch)

        str_epoch = str(epoch + 1).zfill(1)
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)
    print("Run Completed Successfully...")


if __name__ == '__main__':
    run_linear3_model()
