import hashlib
import json
import math
from typing import Type

import torch.backends.cudnn
from torch import nn
from torch.nn import Flatten

from dataloaders.load_vision_dataset import load_vision_dataset
from nn.activations.Activation import Activation
from nn.backward_pass.BackwardFunction import BackwardUsingForward
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
from nn.utils.summary import summary
from utils.data_dirs import data_dirs
from utils.helper_functions import pick_instanceof
from utils.path_functions import path_join
from utils.save_graph import save_graph


def cross_entropy_loss_accuracy(output, target):
    _, preds = torch.max(output.data, 1)
    correct = (preds == target).sum().item()
    return correct / len(output)


class ConvLinearModel(FullSequential):
    def __init__(
            self,
            input_shape,
            conv_features_sizes, linear_features_sizes,
            activation_class: Type[Activation],
            approach: str = "default",
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(ConvLinearModel, self).__init__()

        self.approach = approach
        self.input_shape = input_shape
        self.conv_features_sizes = conv_features_sizes
        self.linear_features_sizes = linear_features_sizes
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage
        self.num_conv_layer = len(conv_features_sizes)
        self.num_linear_layer = len(linear_features_sizes) - 1

        temp_x = torch.zeros(input_shape, requires_grad=False)

        self.all_layers = []
        for i in range(len(conv_features_sizes)):
            conv_layer = nn.Conv2d(
                in_channels=conv_features_sizes[i][0],
                out_channels=conv_features_sizes[i][1],
                kernel_size=conv_features_sizes[i][2]
            )
            self.all_layers.append(BackwardWrapper(conv_layer))
            temp_x = conv_layer(temp_x)

            if i < len(conv_features_sizes) - 1:
                max_pool = nn.MaxPool2d(2, 2)
                self.all_layers.append(BackwardWrapper(max_pool))
                temp_x = max_pool(temp_x)

        flatten = Flatten(start_dim=1)
        self.all_layers.append(BackwardWrapper(flatten))
        temp_x = flatten(temp_x)

        for i in range(len(linear_features_sizes)):
            linear_layer = Linear(in_features=temp_x.shape[1], out_features=linear_features_sizes[i])
            temp_x = linear_layer(temp_x)

            if activation_class is not None:
                activation_class.initialise_(linear_layer.weight)

            if norm_class is not None:
                self.all_layers.append(norm_class())
            if precision_class is not None:
                self.all_layers.append(precision_class(precision=precision))
            if noise_class is not None:
                self.all_layers.append(noise_class(leakage=leakage, precision=precision))

            self.all_layers.append(linear_layer)

            if noise_class is not None:
                self.all_layers.append(noise_class(leakage=leakage, precision=precision))
            if norm_class is not None:
                self.all_layers.append(norm_class())
            if precision_class is not None:
                self.all_layers.append(precision_class(precision=precision))

            if activation_class is not None:
                self.all_layers.append(activation_class())

        self.conv2d_layers = pick_instanceof(self.all_layers, nn.Conv2d)
        self.max_pool2d_layers = pick_instanceof(self.all_layers, nn.MaxPool2d)
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

        self.add_sequence(*self.all_layers)

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
            'num_conv_layer': self.num_conv_layer,
            'num_linear_layer': self.num_linear_layer,
            'conv_features_sizes': self.conv_features_sizes,
            'linear_features_sizes': self.linear_features_sizes,
            'approach': self.approach,
            'activation_class': self.activation_class.__name__ if self.activation_class is not None else str(None),
            'norm_class_y': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_y': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_y': self.precision,
            'noise_class_y': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_y': self.leakage,

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

    def hyperparameters(self):
        return {
            'weight_model_class': self.__class__.__name__,

            'norm_class_w': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_w': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_w': self.precision,
            'noise_class_w': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_w': self.leakage,
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
        is_cuda=is_cuda
    )

    nn_model_params = parameters.nn_model_params
    weight_model_params = parameters.weight_model_params
    nn_model_params["input_shape"] = input_shape
    nn_model_params["conv_features_sizes"] = CONV_LAYER_SIZES[parameters.num_conv_layer]
    nn_model_params["linear_features_sizes"] = LINEAR_LAYER_SIZES[parameters.num_linear_layer] + [len(classes)]
    del nn_model_params["num_conv_layer"]
    del nn_model_params["num_linear_layer"]

    print(f"Creating Models...")
    nn_model = ConvLinearModel(**nn_model_params)
    weight_model = WeightModel(**weight_model_params)
    if parameters.tensorboard:
        nn_model.create_tensorboard(paths.tensorboard)

    nn_model.compile(device=device, layer_data=True)
    nn_model.loss_fn = nn.CrossEntropyLoss()
    nn_model.accuracy_fn = cross_entropy_loss_accuracy
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
        train_loss, train_accuracy = nn_model.train_on(
            train_loader,
            epoch=epoch,
            test_run=parameters.test_run,
            parameters_to_apply_fn=[PseudoOptimizer.parameter_type.update_params]
        )
        test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch, test_run=parameters.test_run)

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

        with open(log_file, "a+") as file:
            file.write(print_str)

        if parameters.save_data:
            torch.save(nn_model.state_dict(), f"{paths.model_data}/{epoch}_state_dict_nn_model")
            torch.save(weight_model.state_dict(), f"{paths.model_data}/{epoch}_state_dict_weight_model")

        if parameters.save_data and epoch == (parameters.epochs - 1):
            torch.save(str(nn_model), f"{paths.model_data}/{epoch}_str_nn_model")
            torch.save(str(weight_model), f"{paths.model_data}/{epoch}_str_weight_model")

            torch.save(parameter_log, f"{paths.model_data}/{epoch}_parameter_log")
            torch.save(loss_accuracy, f"{paths.model_data}/{epoch}_loss_accuracy")

            torch.save(nn_model.hyperparameters(), f"{paths.model_data}/{epoch}_hyperparameters_nn_model")
            torch.save(weight_model.hyperparameters(), f"{paths.model_data}/{epoch}_hyperparameters_weight_model")

            torch.save(nn_model.optimizer.state_dict(), f"{paths.model_data}/{epoch}_state_dict_optimizer")

        if parameters.tensorboard and epoch == (parameters.epochs - 1):
            parameter_log["input_shape"] = "_".join([str(x) for x in parameter_log["input_shape"]])
            parameter_log["conv_features_sizes"] = "_".join([str(x) for x in parameter_log["conv_features_sizes"]])
            parameter_log["linear_features_sizes"] = "_".join([str(x) for x in parameter_log["linear_features_sizes"]])
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
