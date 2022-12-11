import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Type, List

import numpy as np
import torch.backends.cudnn
import torchvision
from torch import optim
from torch.nn import Flatten
from torch.optim import Optimizer
from torchvision.datasets import VisionDataset

from _research.crc._common import pick_instanceof_module
from _research.crc.analog_vnn_1_model import WeightModel
from _research.dataloaders.load_vision_dataset import load_vision_dataset
from _research.utils.data_dirs import data_dirs
from _research.utils.path_functions import path_join
from _research.utils.save_graph import save_graph
from nn.layers.BackwardWrapper import BackwardWrapper
from nn.layers.Linear import Linear
from nn.layers.activations.Activation import Activation
from nn.layers.functionals.Normalize import *
from nn.layers.functionals.ReducePrecision import ReducePrecision
from nn.layers.functionals.StochasticReducePrecision import StochasticReducePrecision
from nn.layers.noise.GaussianNoise import GaussianNoise
from nn.modules.FullSequential import FullSequential
from nn.optimizer.PseudoOptimizer import PseudoOptimizer
from nn.utils.is_cpu_cuda import is_cpu_cuda
from nn.utils.summary import summary


@dataclass
class RunParametersParneet:
    name: Union[None, str] = None
    data_folder: Union[None, str] = None

    activation_class: Union[None, Type[Activation]] = None
    norm_class: Union[None, Type[Normalize]] = None
    precision_class: Type[Layer] = None
    precision: Union[None, int] = None
    noise_class: Type[Layer] = None
    leakage: Union[None, float] = None

    w_norm_class: Union[None, Type[Normalize]] = None
    w_precision_class: Type[Layer] = None
    w_precision: Union[None, int] = None
    w_noise_class: Type[Layer] = None
    w_leakage: Union[None, float] = None

    optimiser_class: Type[Optimizer] = optim.Adam
    optimiser_parameters: dict = None
    dataset: Type[VisionDataset] = torchvision.datasets.CIFAR10
    color: bool = True
    batch_size: int = 128
    epochs: int = 200

    device: Union[None, torch.device] = None
    test_logs: bool = False
    test_run: bool = False
    tensorboard: bool = False
    save_data: bool = True
    timestamp: str = None

    def __init__(self):
        self.optimiser_parameters = {}

    @property
    def nn_model_params(self):
        return {
            "activation_class": self.activation_class,
            "norm_class": self.norm_class,
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


class ParneetModel(FullSequential):
    def __init__(
            self,
            input_shape, num_classes,
            activation_class: Type[Activation],
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(ParneetModel, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers: List[nn.Module] = []

        self.add_doa_layers()
        self.all_layers.append(BackwardWrapper(nn.Conv2d(
            in_channels=self.input_shape[1],
            out_channels=48,
            kernel_size=(3, 3),
            padding=(1, 1)
        )))
        self.add_aod_layers()
        self.add_doa_layers()
        self.all_layers.append(BackwardWrapper(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3))))
        self.add_aod_layers()
        self.all_layers.append(BackwardWrapper(nn.MaxPool2d(2, 2)))
        self.all_layers.append(BackwardWrapper(nn.Dropout(0.25)))
        self.add_doa_layers()
        self.all_layers.append(
            BackwardWrapper(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1))))
        self.add_aod_layers()
        self.add_doa_layers()
        self.all_layers.append(BackwardWrapper(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3))))
        self.add_aod_layers()
        self.all_layers.append(BackwardWrapper(nn.MaxPool2d(2, 2)))
        self.all_layers.append(BackwardWrapper(nn.Dropout(0.25)))
        self.add_doa_layers()
        self.all_layers.append(
            BackwardWrapper(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1))))
        self.add_aod_layers()
        self.add_doa_layers()
        self.all_layers.append(BackwardWrapper(nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3))))
        self.add_aod_layers()
        self.all_layers.append(BackwardWrapper(nn.MaxPool2d(2, 2)))
        self.all_layers.append(BackwardWrapper(nn.Dropout(0.25)))

        self.all_layers.append(BackwardWrapper(Flatten(start_dim=1)))

        self.add_doa_layers()
        self.all_layers.append(Linear(in_features=self.get_in_shape(input_shape)[1], out_features=512))
        self.add_aod_layers()
        self.all_layers.append(BackwardWrapper(nn.Dropout(0.5)))
        self.add_doa_layers()
        self.all_layers.append(Linear(in_features=512, out_features=256))
        self.add_aod_layers()
        self.all_layers.append(BackwardWrapper(nn.Dropout(0.5)))
        self.add_doa_layers()
        self.all_layers.append(Linear(in_features=256, out_features=num_classes))
        self.add_aod_layers()

        self.conv2d_layers = pick_instanceof_module(self.all_layers, nn.Conv2d)
        self.max_pool2d_layers = pick_instanceof_module(self.all_layers, nn.MaxPool2d)
        self.linear_layers = pick_instanceof_module(self.all_layers, Linear)
        self.activation_layers = pick_instanceof_module(self.all_layers, self.activation_class)
        self.norm_layers = pick_instanceof_module(self.all_layers, norm_class)
        self.precision_layers = pick_instanceof_module(self.all_layers, precision_class)
        self.noise_layers = pick_instanceof_module(self.all_layers, noise_class)

        for i in self.linear_layers:
            self.activation_class.initialise_(i.weight)

        self.add_sequence(*self.all_layers)

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

    def get_in_shape(self, input_shape):
        with torch.no_grad():
            temp_x = torch.zeros(input_shape, requires_grad=False)
            for i in self.all_layers:
                # print(temp_x.shape)
                temp_t = i.training
                i.train(False)
                temp_x = i(temp_x)
                i.train(temp_t)
        return temp_x.shape

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

            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'activation_class': self.activation_class.__name__ if self.activation_class is not None else str(None),
            'norm_class_y': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': self.precision,
            'noise_class_y': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_y': self.leakage,

            'loss_class': self.loss_function.__class__.__name__,
            'accuracy_fn': self.accuracy_function.__name__,
            'optimiser_superclass': self.optimizer.__class__.__name__,
        }


def run_parneet_model(parameters: RunParametersParneet):
    print(parameters.color)
    torch.backends.cudnn.benchmark = True

    if parameters.device is not None:
        is_cpu_cuda.set_device(parameters.device)
    device, is_cuda = is_cpu_cuda.is_using_cuda()
    parameters.device = device

    if parameters.data_folder is None:
        raise Exception("data_folder is None")

    if parameters.name is None:
        parameters.name = hashlib.sha256(str(parameters).encode("utf-8")).hexdigest()

    print(f"Parameters: {parameters}")
    print(f"Name: {parameters.name}")
    print(f"Device: {parameters.device}")

    paths = data_dirs(parameters.data_folder, name=parameters.name, timestamp=parameters.timestamp,
                      tensorboard=parameters.tensorboard)
    log_file = path_join(paths.logs, f"{paths.name}_logs.txt")

    print(f"Timestamp: {paths.timestamp}")
    print(f"Storage name: {paths.name}")
    print()

    print(f"Loading Data...")
    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=parameters.dataset,
        path=paths.dataset,
        batch_size=parameters.batch_size,
        is_cuda=is_cuda,
        grayscale=not parameters.color
    )

    nn_model_params = parameters.nn_model_params
    weight_model_params = parameters.weight_model_params
    nn_model_params["input_shape"] = input_shape
    nn_model_params["num_classes"] = len(classes)

    print(f"Creating Models...")
    nn_model = ParneetModel(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        nn_model.create_tensorboard(paths.tensorboard)

    nn_model.compile(device=device, layer_data=True)
    nn_model.loss_function = nn.CrossEntropyLoss()
    nn_model.accuracy_function = cross_entropy_loss_accuracy
    nn_model.to(device=device)
    weight_model.to(device=device)
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
        'color': parameters.color,
        'epochs': parameters.epochs,

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

    if parameters.tensorboard:
        nn_model.tensorboard.tensorboard.add_text("parameters", json.dumps(parameters.json, sort_keys=True, indent=2))
    data: Tensor = next(iter(train_loader))[0]

    print(f"Saving Graphs...")
    save_graph(path_join(paths.logs, f"{paths.name}_nn_model"), nn_model, data)
    save_graph(path_join(paths.logs, f"{paths.name}_weight_model"), weight_model, torch.ones((1, 1)))

    if parameters.tensorboard:
        nn_model.tensorboard.add_graph(train_loader)
        # nn_model.tensorboard.add_graph(train_loader, model=weight_model)

    loss_accuracy = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print(f"Starting Training...")
    for epoch in range(parameters.epochs):
        if parameters.test_logs:
            break

        train_loss, train_accuracy = nn_model.train_on(
            train_loader,
            epoch=epoch,
            test_run=parameters.test_run,
            parameters_to_apply_fn=[PseudoOptimizer.parameter_type.update_params]
        )
        test_loss, test_accuracy = nn_model.test_on(
            test_loader,
            epoch=epoch,
            test_run=parameters.test_run
        )

        loss_accuracy["train_loss"].append(train_loss)
        loss_accuracy["train_accuracy"].append(train_accuracy)
        loss_accuracy["test_loss"].append(test_loss)
        loss_accuracy["test_accuracy"].append(test_accuracy)

        str_epoch = str(epoch + 1).zfill(math.ceil(math.log10(parameters.epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        parameter_log["last_epoch"] = epoch
        with open(log_file, "a+") as file:
            file.write(print_str)

        if parameters.save_data:
            torch.save(nn_model.state_dict(), f"{paths.model_data}/{epoch}_state_dict_nn_model")
            torch.save(weight_model.state_dict(), f"{paths.model_data}/{epoch}_state_dict_weight_model")

        if train_accuracy < 0.125 and epoch >= 9:
            break

        if parameters.test_run:
            break

    if parameters.save_data:
        torch.save(str(nn_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_nn_model")
        torch.save(str(weight_model), f"{paths.model_data}/{parameter_log['last_epoch']}_str_weight_model")

        torch.save(parameters.json, f"{paths.model_data}/{parameter_log['last_epoch']}_parameters_json")
        torch.save(parameter_log, f"{paths.model_data}/{parameter_log['last_epoch']}_parameter_log")
        torch.save(loss_accuracy, f"{paths.model_data}/{parameter_log['last_epoch']}_loss_accuracy")

        torch.save(nn_model.hyperparameters(),
                   f"{paths.model_data}/{parameter_log['last_epoch']}_hyperparameters_nn_model")
        torch.save(weight_model.hyperparameters(),
                   f"{paths.model_data}/{parameter_log['last_epoch']}_hyperparameters_weight_model")

        torch.save(nn_model.optimizer.state_dict(),
                   f"{paths.model_data}/{parameter_log['last_epoch']}_state_dict_optimizer")

    if parameters.tensorboard:
        parameter_log["input_shape"] = "_".join([str(x) for x in parameter_log["input_shape"]])
        metric_dict = {
            "train_loss": loss_accuracy["train_loss"][-1],
            "train_accuracy": loss_accuracy["train_accuracy"][-1],
            "test_loss": loss_accuracy["test_loss"][-1],
            "test_accuracy": loss_accuracy["test_accuracy"][-1],
            "min_train_loss": np.min(loss_accuracy["train_loss"]),
            "max_train_accuracy": np.max(loss_accuracy["train_accuracy"]),
            "min_test_loss": np.min(loss_accuracy["test_loss"]),
            "max_test_accuracy": np.max(loss_accuracy["test_accuracy"]),
        }
        nn_model.tensorboard.tensorboard.add_hparams(
            hparam_dict=parameter_log,
            metric_dict=metric_dict
        )

    with open(log_file, "a+") as file:
        file.write("Run Completed Successfully...")
    print()
