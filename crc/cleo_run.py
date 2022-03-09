import json
import math
from typing import Type, Union

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.nn import Flatten

from cleo_runs.common_parameter_fn import normalize_parameter, add_gaussian_noise
from dataloaders.load_vision_dataset import load_vision_dataset
from nn.activations.Activation import Activation
from nn.activations.Identity import Identity
from nn.activations.ReLU import ReLU
from nn.backward_pass.BackwardFunction import BackwardIdentity, BackwardUsingForward
from nn.layers.GaussianNoise import GaussianNoise
from nn.layers.Linear import Linear, LinearBackpropagation
from nn.layers.Normalize import Normalize
from nn.layers.ReducePrecision import ReducePrecision
from nn.layers.StochasticReducePrecision import StochasticReducePrecision
from nn.modules.FullSequential import FullSequential
from nn.optimizer.BaseOptimizer import BaseOptimizer
from nn.optimizer.IdentityOptimizer import IdentityOptimizer
from nn.optimizer.ReducePrecisionOptimizer import ReducePrecisionOptimizer
from nn.optimizer.StochasticReducePrecisionOptimizer import StochasticReducePrecisionOptimizer
from nn.utils.is_using_cuda import get_device, is_using_cuda
from nn.utils.make_dot import make_dot
from nn.utils.summary import summary
from utils.data_dirs import data_dirs
from utils.helper_functions import pick_instanceof
from utils.path_functions import path_join

TENSORBOARD = True
LAYER_SIZES = {
    2: [128],
    3: [256, 64],
    4: [256, 128, 64]
}


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename, format="svg", cleanup=True)


def cross_entropy_loss_accuracy(output, target):
    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return correct / len(output)


class LinearModel(FullSequential):
    def __init__(
            self, layer_features_sizes,
            activation_class: Type[Activation],
            approach: str = "default",
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(LinearModel, self).__init__()

        self.approach = approach
        self.layer_features_sizes = layer_features_sizes
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage
        self.num_layer = len(layer_features_sizes) - 1

        self.all_layers = []

        for i in range(1, len(layer_features_sizes)):
            linear_layer = Linear(in_features=layer_features_sizes[i - 1], out_features=layer_features_sizes[i])
            linear_layer.use(LinearBackpropagation)
            activation_class.initialise_(linear_layer.weight)

            self.all_layers += [
                norm_class() if norm_class is not None else Identity("Norm"),
                precision_class(precision=precision) if precision_class is not None else Identity("Precision"),
                noise_class(leakage=leakage, precision=precision) if noise_class is not None else Identity("Noise"),

                linear_layer,

                noise_class(leakage=leakage, precision=precision) if noise_class is not None else Identity("Noise"),
                norm_class() if norm_class is not None else Identity("Norm"),
                precision_class(precision=precision) if precision_class is not None else Identity("Precision"),

                activation_class(),
            ]

        self.linear_layers = pick_instanceof(self.all_layers, Linear)
        self.activation_layers = pick_instanceof(self.all_layers, activation_class)
        self.norm_layers = pick_instanceof(self.all_layers, norm_class)
        self.precision_layers = pick_instanceof(self.all_layers, precision_class)
        self.noise_layers = pick_instanceof(self.all_layers, noise_class)

        if approach == "default":
            pass
        if approach == "use_autograd_graph":
            self.backward.use_autograd_graph = True
        if approach == "no_norm_grad":
            [i.use(BackwardIdentity) for i in self.norm_layers]
        if approach == "norm_grad_by_forward":
            [i.use(BackwardUsingForward) for i in self.norm_layers]

        self.add_sequence(
            Flatten(start_dim=1),
            self.backward.STOP,
            *self.all_layers
        )

        if precision_class == ReducePrecision or precision_class == StochasticReducePrecision:
            precision_class.parameter_class.convert_model(
                self,
                precision=precision,
            )

    @property
    def optimizer_superclass(self) -> Type[BaseOptimizer]:
        if self.precision_class == ReducePrecision:
            return ReducePrecisionOptimizer
        elif self.precision_class == StochasticReducePrecision:
            return StochasticReducePrecisionOptimizer
        else:
            return IdentityOptimizer

    def set_optimizer(self, optimizer_cls, **optimiser_parameters):
        self.optimizer = self.optimizer_superclass(
            optimizer_cls=optimizer_cls,
            params=self.parameters(),
            **optimiser_parameters
        )
        return self

    def hyperparameters(self):
        return {
            'model_class': self.__class__.__name__,

            'num_layer': self.num_layer,
            'layer_features_sizes': self.layer_features_sizes,
            'approach': self.approach,
            'activation_class': self.activation_class.__name__,
            'norm_class_y': self.norm_class.__name__ if self.noise_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': str(self.precision),
            'noise_class_y': self.noise_class.__name__ if self.precision_class is not None else str(None),
            'leakage_y': str(self.leakage),

            'loss_class': self.loss_fn.__class__.__name__,
            'accuracy_fn': self.accuracy_fn.__name__,
            'optimiser_superclass': self.optimizer.__class__.__name__,
        }


def main(
        name, data_folder, model_params,
        norm_class_w, precision_w, leakage_w,
        optimiser_class, optimiser_parameters,
        dataset, batch_size, epochs
):
    device, is_cuda = is_using_cuda()
    torch.manual_seed(0)
    print()
    print(device, name)

    paths = data_dirs(data_folder, name=name)
    log_file = path_join(paths.logs, f"{paths.name}_logs.txt")

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=dataset,
        path=paths.dataset,
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    model_params["layer_features_sizes"] = [int(np.prod(input_shape[1:]))] + LAYER_SIZES[model_params["num_layer"]] + [len(classes)]
    del model_params["num_layer"]
    model = LinearModel(**model_params)
    if TENSORBOARD:
        model.create_tensorboard(paths.tensorboard)

    model.compile(device=get_device(), layer_data=True)
    model.loss_fn = nn.CrossEntropyLoss()
    model.accuracy_fn = cross_entropy_loss_accuracy
    model.set_optimizer(optimizer_cls=optimiser_class, **optimiser_parameters)

    parameter_log = {
        'dataset': dataset.__name__,
        'batch_size': batch_size,
        'is_cuda': is_cuda,

        **model.hyperparameters(),

        # 'norm_class_w': norm_class_w.__name__,
        # 'precision_class_w': self.precision_class.__name__ if self.precision_class is not None else str(None),
        # 'precision_w': str(precision_w),
        # 'noise_class_w': add_gaussian_noise.__name__ if leakage_w is not None else str(None),
        # 'leakage_w': str(leakage_w),
    }

    with open(log_file, "a+") as file:
        file.write(json.dumps(parameter_log, sort_keys=True, indent=2) + "\n\n")
        file.write(str(model.optimizer) + "\n\n")

        file.write(str(model) + "\n\n")
        file.write(summary(model, input_size=tuple(input_shape[1:])) + "\n\n")

    data = next(iter(train_loader))[0]
    data = data.to(model.device)
    save_graph(path_join(paths.logs, paths.name), model(data), model.named_parameters())

    # apply_fn = [normalize_parameter(norm_class_w), add_gaussian_noise(leakage_w, precision_w)]
    apply_fn = None
    for epoch in range(epochs):
        train_loss, train_accuracy = model.train_on(train_loader, epoch=epoch, apply_fn=apply_fn)
        test_loss, test_accuracy = model.test_on(test_loader, epoch=epoch)

        str_epoch = str(epoch + 1).zfill(math.ceil(math.log10(epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        with open(log_file, "a+") as file:
            file.write(print_str)

        if TENSORBOARD and epoch == epochs:
            model.tensorboard.tensorboard.add_hparams(
                hparam_dict=parameter_log,
                metric_dict={
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )
    print()


if __name__ == '__main__':
    main(
        name="test",
        data_folder="C:/_data/cleo_test",
        model_params={
            "num_layer": 2,
            "activation_class": ReLU,
            "approach": "default",
            "norm_class": None,
            "precision_class": None,
            "precision": None,
            "noise_class": None,
            "leakage": None,
        },
        norm_class_w=None,
        precision_w=None,
        leakage_w=None,
        optimiser_class=optim.Adam,
        optimiser_parameters={},
        dataset=torchvision.datasets.MNIST,
        batch_size=128,
        epochs=1
    )
