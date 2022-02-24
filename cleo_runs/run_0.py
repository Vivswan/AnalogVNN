import json
import math
from typing import Type

import numpy as np
from torch import nn
from torch.nn import Flatten

from cleo_runs.common import *
from dataloaders.load_vision_dataset import load_vision_dataset
from nn.activations.Activation import Activation
from nn.layers.Linear import Linear, LinearBackpropagation
from nn.modules.FullSequential import FullSequential
from nn.utils.is_using_cuda import get_device, is_using_cuda
from nn.utils.summary import summary
from utils.data_dirs import data_dirs
from utils.path_functions import path_join


class Linear2(FullSequential):
    def __init__(self, approach: str, in_features, out_features, activation_class: Type[Activation]):
        super(Linear2, self).__init__()
        self.activation_class = activation_class

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=128).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.activation1 = activation_class()
        self.activation2 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)

        if approach == "default":
            self.backward.use_autograd_graph = True
        if approach == "full":
            pass

        self.set_full_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.linear1,
            self.activation1,

            self.linear2,
            self.activation2,
        )


class Linear3(FullSequential):
    def __init__(self, approach: str, in_features, out_features, activation_class: Type[Activation]):
        super(Linear3, self).__init__()
        self.activation_class = activation_class

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=256).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=64).use(LinearBackpropagation)
        self.linear3 = Linear(in_features=self.linear2.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.activation1 = activation_class()
        self.activation2 = activation_class()
        self.activation3 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)
        activation_class.initialise_(self.linear3.weight)

        if approach == "default":
            self.backward.use_autograd_graph = True
        if approach == "full":
            pass

        self.set_full_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.linear1,
            self.activation1,

            self.linear2,
            self.activation2,

            self.linear3,
            self.activation3,
        )


class Linear4(FullSequential):
    def __init__(self, approach: str, in_features, out_features, activation_class: Type[Activation]):
        super(Linear4, self).__init__()
        self.activation_class = activation_class

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=256).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=128).use(LinearBackpropagation)
        self.linear3 = Linear(in_features=self.linear2.out_features, out_features=64).use(LinearBackpropagation)
        self.linear4 = Linear(in_features=self.linear3.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.activation1 = activation_class()
        self.activation2 = activation_class()
        self.activation3 = activation_class()
        self.activation4 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)
        activation_class.initialise_(self.linear3.weight)
        activation_class.initialise_(self.linear4.weight)

        if approach == "default":
            self.backward.use_autograd_graph = True
        if approach == "full":
            pass

        self.set_full_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.linear1,
            self.activation1,

            self.linear2,
            self.activation2,

            self.linear3,
            self.activation3,

            self.linear4,
            self.activation4,
        )


def main(
        name, data_folder,
        model_class: Type[Linear2],
        approach, activation_class,
        optimiser_class, optimiser_parameters,
        dataset, batch_size, epochs
):
    paths = data_dirs(data_folder, name=name)
    device, is_cuda = is_using_cuda()
    print(device, name)
    log_file = path_join(paths.logs, f"{paths.name}_logs.txt")

    train_loader, test_loader, input_shape, classes = load_vision_dataset(
        dataset=dataset,
        path=paths.dataset,
        batch_size=batch_size,
        is_cuda=is_cuda
    )

    model = model_class(
        approach=approach,
        in_features=input_shape[1:],
        out_features=len(classes),
        activation_class=activation_class,
    )
    if TENSORBOARD:
        model.create_tensorboard(paths.tensorboard)

    model.compile(device=get_device(), layer_data=True)
    model.loss_fn = nn.CrossEntropyLoss()
    model.accuracy_fn = cross_entropy_loss_accuracy
    model.optimizer = optimiser_class(params=model.parameters(), **optimiser_parameters)

    parameter_log = {
        "model": model_class.__name__,
        "approach": approach,
        "dataset": dataset.__name__,
        "activation_class": model.activation_class.__name__,
        "loss_class": model.loss_fn.__class__.__name__,
        "model_class": model_class.__name__,
        'optimiser_class': optimiser_class.__name__,
        'batch_size': batch_size
    }

    with open(log_file, "a+") as file:
        file.write(json.dumps(parameter_log, sort_keys=True, indent=2) + "\n\n")
        file.write(str(model.optimizer) + "\n\n")

        file.write(str(model) + "\n\n")
        file.write(summary(model, input_size=tuple(input_shape[1:])) + "\n\n")

    for epoch in range(epochs):
        train_loss, train_accuracy = model.train_on(train_loader, epoch=epoch)
        test_loss, test_accuracy = model.test_on(test_loader, epoch=epoch)

        str_epoch = str(epoch).zfill(math.ceil(math.log10(epochs)))
        print_str = f'({str_epoch})' \
                    f' Training Loss: {train_loss:.4f},' \
                    f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                    f' Testing Loss: {test_loss:.4f},' \
                    f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
        print(print_str)

        with open(log_file, "a+") as file:
            file.write(print_str)

    if TENSORBOARD:
        model.tensorboard.tensorboard.add_hparams(
            hparam_dict=parameter_log,
            metric_dict={
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

    data = next(iter(train_loader))[0]
    data = data.to(model.device)
    save_graph(path_join(paths.logs, paths.name), model(data), model.named_parameters())
    print()
