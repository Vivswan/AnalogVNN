import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Type

import numpy as np
import torch.backends.cudnn
import torchvision
from torch import nn, optim
from torch.nn import Flatten
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.activations.Activation import Activation
from nn.activations.Identity import Identity
from nn.backward_pass.BackwardFunction import BackwardUsingForward
from nn.layers.BackwardWrapper import BackwardWrapper
from nn.layers.Linear import Linear, LinearBackpropagation
from nn.layers.Normalize import *
from nn.layers.ReducePrecision import ReducePrecision
from nn.layers.StochasticReducePrecision import StochasticReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.modules.FullSequential import FullSequential
from nn.modules.Sequential import Sequential
from nn.optimizer.PseudoOptimizer import PseudoOptimizer
from nn.utils.is_cpu_cuda import is_cpu_cuda
from nn.utils.summary import summary
from utils.data_dirs import data_dirs
from utils.helper_functions import pick_instanceof
from utils.path_functions import path_join
from utils.save_graph import save_graph

LAYER_SIZES = {
    1: [],
    2: [128],
    3: [256, 64],
    4: [256, 128, 64]
}


@dataclass
class RunParameters:
    name: Union[None, str] = None
    data_folder: Union[None, str] = None

    num_layer: Union[None, int] = 1
    activation_class: Union[None, Type[Activation]] = None
    norm_class: Union[None, Type[Normalize]] = None
    approach: Union[None, str] = "default"
    precision_class: Type[BaseLayer] = None
    precision: Union[None, int] = None
    noise_class: Type[BaseLayer] = None
    leakage: Union[None, float] = None

    w_norm_class: Union[None, Type[Normalize]] = None
    w_precision_class: Type[BaseLayer] = None
    w_precision: Union[None, int] = None
    w_noise_class: Type[BaseLayer] = None
    w_leakage: Union[None, float] = None

    optimiser_class: Type[Optimizer] = optim.Adam
    optimiser_parameters: dict = None
    dataset: Type[VisionDataset] = torchvision.datasets.MNIST
    batch_size: int = 1280
    epochs: int = 10

    device: Union[None, torch.device] = None
    test_run: bool = False
    tensorboard: bool = True

    def __init__(self):
        self.optimiser_parameters = {}

    @property
    def nn_model_params(self):
        return {
            "num_layer": self.num_layer,
            "activation_class": self.activation_class,
            "norm_class": self.norm_class,
            "approach": self.approach,
            "precision_class": self.precision_class,
            "precision": self.precision,
            "noise_class": self.noise_class,
            "leakage": self.leakage,
        }

    @property
    def weight_model_params(self):
        return {
            "norm_class": self.w_norm_class,
            "precision_class": self.w_precision_class,
            "precision": self.w_precision,
            "noise_class": self.w_noise_class,
            "leakage": self.w_leakage,
        }

    @property
    def json(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def __repr__(self):
        return f"RunParameters({json.dumps(self.json)})"


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
            if activation_class is not None:
                activation_class.initialise_(linear_layer.weight)

            self.all_layers += [
                norm_class() if norm_class is not None else Identity("Norm"),
                precision_class(precision=precision) if precision_class is not None else Identity("Precision"),
                noise_class(leakage=leakage, precision=precision) if noise_class is not None else Identity("Noise"),

                linear_layer,

                noise_class(leakage=leakage, precision=precision) if noise_class is not None else Identity("Noise"),
                norm_class() if norm_class is not None else Identity("Norm"),
                precision_class(precision=precision) if precision_class is not None else Identity("Precision"),

                activation_class() if activation_class is not None else Identity("Activation"),
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
            BackwardWrapper(Flatten(start_dim=1)),
            *self.all_layers
        )

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

    def hyperparameters(self):
        return {
            'nn_model_class': self.__class__.__name__,

            'num_layer': self.num_layer,
            'layer_features_sizes': self.layer_features_sizes,
            'approach': self.approach,
            'activation_class': self.activation_class.__name__ if self.activation_class is not None else str(None),
            'norm_class_y': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': str(self.precision),
            'noise_class_y': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_y': str(self.leakage),

            'loss_class': self.loss_fn.__class__.__name__,
            'accuracy_fn': self.accuracy_fn.__name__,
            'optimiser_superclass': self.optimizer.__class__.__name__,
        }


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

        self.norm_layer = norm_class() if norm_class is not None else Identity(name="Norm")
        self.precision_layer = precision_class(precision=precision) if precision_class is not None else Identity(
            name="Precision")
        self.noise_layer = noise_class(leakage=leakage, precision=precision) if noise_class is not None else Identity(
            name="Noise")

        self.eval()
        self.add_sequence(self.norm_layer, self.precision_layer, self.noise_layer)

    def hyperparameters(self):
        return {
            'weight_model_class': self.__class__.__name__,

            'norm_class_w': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_w': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_w': str(self.precision),
            'noise_class_w': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_w': str(self.leakage),
        }


def run_main_model(parameters: RunParameters):
    torch.backends.cudnn.benchmark = True

    if parameters.device is not None:
        is_cpu_cuda.set_device(parameters.device)
    device, is_cuda = is_cpu_cuda.is_using_cuda()
    parameters.device = device

    if parameters.data_folder is None:
        raise Exception("data_folder is None")

    if parameters.name is None:
        parameters.name = hashlib.sha256(str(parameters).encode("utf-8")).hexdigest()

    print(parameters)
    print(parameters.device, parameters.name)

    # return

    paths = data_dirs(parameters.data_folder, name=parameters.name)
    log_file = path_join(paths.logs, f"{paths.name}_logs.txt")

    print(f"Loading Data...")
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=parameters.dataset,
        path=paths.dataset,
        batch_size=parameters.batch_size,
        is_cuda=is_cuda
    )

    nn_model_params = parameters.nn_model_params
    weight_model_params = parameters.weight_model_params
    nn_model_params["layer_features_sizes"] = [int(np.prod(input_shape[1:]))] + LAYER_SIZES[parameters.num_layer] + [
        len(classes)]
    del nn_model_params["num_layer"]

    print(f"Creating Models...")
    nn_model = LinearModel(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        nn_model.create_tensorboard(paths.tensorboard)

    nn_model.compile(device=device, layer_data=True)
    nn_model.loss_fn = nn.CrossEntropyLoss()
    nn_model.accuracy_fn = cross_entropy_loss_accuracy
    PseudoOptimizer.parameter_type.convert_model(nn_model, transform=weight_model)
    nn_model.set_optimizer(
        super_optimizer_cls=PseudoOptimizer,
        optimizer_cls=parameters.optimiser_class,
        **parameters.optimiser_parameters
    )

    parameter_log = {
        'dataset': parameters.dataset.__name__,
        'batch_size': parameters.batch_size,
        'is_cuda': is_cuda,

        **nn_model.hyperparameters(),
        **weight_model.hyperparameters(),
    }

    print(f"Creating Log File...")
    with open(log_file, "a+") as file:
        file.write(json.dumps(parameters.json, sort_keys=True, indent=2) + "\n\n")
        file.write(json.dumps(parameter_log, sort_keys=True, indent=2) + "\n\n")
        file.write(str(nn_model.optimizer) + "\n\n")

        file.write(str(nn_model) + "\n\n")
        file.write(str(weight_model) + "\n\n")
        file.write(summary(nn_model, input_size=tuple(input_shape[1:])) + "\n\n")
        file.write(summary(weight_model, input_size=(1, 1)) + "\n\n")

    nn_model.tensorboard.tensorboard.add_text("parameters", json.dumps(parameters.json, sort_keys=True, indent=2))
    data: Tensor = next(iter(train_loader))[0]

    print(f"Saving Graphs...")
    save_graph(path_join(paths.logs, f"{paths.name}_nn_model"), nn_model, data)
    save_graph(path_join(paths.logs, f"{paths.name}_weight_model"), weight_model, torch.ones((1, 1)))
    nn_model.tensorboard.add_graph(train_loader)
    # nn_model.tensorboard.add_graph(train_loader, model=weight_model)

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
        train_loss, train_accuracy = nn_model.train_on(
            train_loader,
            epoch=epoch,
            test_run=parameters.test_run,
            parameters_to_apply_fn=[PseudoOptimizer.parameter_type.update_params]
        )
        test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch, test_run=parameters.test_run)

        str_epoch = str(epoch + 1).zfill(math.ceil(math.log10(parameters.epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        with open(log_file, "a+") as file:
            file.write(print_str)

        if parameters.tensorboard and epoch == parameters.epochs:
            nn_model.tensorboard.tensorboard.add_hparams(
                hparam_dict=parameter_log,
                metric_dict={
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )

        if parameters.test_run:
            break

    with open(log_file, "a+") as file:
        file.write("Run Completed Successfully...")
    print()
