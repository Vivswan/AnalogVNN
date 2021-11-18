import json
import math
from typing import Type

import numpy as np
from torch import nn
from torch.nn import Flatten

from cleo_runs.common import *
from cleo_runs.common_parameter_fn import normalize_parameter, add_gaussian_noise
from dataloaders.load_vision_dataset import load_vision_dataset
from nn.BaseModel import BaseModel
from nn.activations.Activation import Activation
from nn.backward_pass.BackwardFunction import BackwardIdentity, BackwardUsingForward
from nn.layers.GaussianNoise import GaussianNoise
from nn.layers.Linear import Linear, LinearBackpropagation
from nn.layers.Normalize import Normalize
from nn.layers.ReducePrecision import ReducePrecision
from nn.optimizer.StochasticReducePrecisionOptimizer import StochasticReducePrecisionOptimizer
from nn.parameters.StochasticReducePrecisionParameter import StochasticReducePrecisionParameter
from nn.utils.is_using_cuda import get_device, is_using_cuda, set_device
from nn.utils.summary import summary
from utils.data_dirs import data_dirs
from utils.path_functions import path_join


class Linear2(BaseModel):
    def __init__(
            self, approach: str, in_features, out_features,
            activation_class: Type[Activation], norm_class: Type[Normalize], precision: int, leakage: float
    ):
        super(Linear2, self).__init__()
        self.activation_class = activation_class
        self.precision = precision

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=128).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.norm1_pre = norm_class()
        self.norm1_post = norm_class()
        self.norm2_pre = norm_class()
        self.norm2_post = norm_class()

        self.rp1_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp1_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_post = ReducePrecision(precision=precision).use(BackwardIdentity)

        self.noise1_pre = GaussianNoise(leakage=leakage, precision=precision)
        self.noise1_post = GaussianNoise(leakage=leakage, precision=precision)
        self.noise2_pre = GaussianNoise(leakage=leakage, precision=precision)
        self.noise2_post = GaussianNoise(leakage=leakage, precision=precision)

        self.activation1 = activation_class()
        self.activation2 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)

        if approach == "full":
            pass
        if approach == "no_norm":
            self.norm1_pre.use(BackwardIdentity)
            self.norm1_post.use(BackwardIdentity)
            self.norm2_pre.use(BackwardIdentity)
            self.norm2_post.use(BackwardIdentity)
        if approach == "use_norm_forward":
            self.norm1_pre.use(BackwardUsingForward)
            self.norm1_post.use(BackwardUsingForward)
            self.norm2_pre.use(BackwardUsingForward)
            self.norm2_post.use(BackwardUsingForward)

        self.add_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.norm1_pre,
            self.rp1_pre,
            self.noise1_pre,
            self.linear1,
            self.noise1_post,
            self.norm1_post,
            self.rp1_post,
            self.activation1,

            self.norm2_pre,
            self.rp2_pre,
            self.noise2_pre,
            self.linear2,
            self.noise2_post,
            self.norm2_post,
            self.rp2_post,
            self.activation2,
        )


class Linear3(BaseModel):
    def __init__(
            self, approach: str, in_features, out_features,
            activation_class: Type[Activation], norm_class: Type[Normalize], precision: int, leakage: float
    ):
        super(Linear3, self).__init__()
        self.activation_class = activation_class
        self.precision = precision

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=256).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=64).use(LinearBackpropagation)
        self.linear3 = Linear(in_features=self.linear2.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.norm1_pre = norm_class()
        self.norm1_post = norm_class()
        self.norm2_pre = norm_class()
        self.norm2_post = norm_class()
        self.norm3_pre = norm_class()
        self.norm3_post = norm_class()

        self.rp1_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp1_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp3_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp3_post = ReducePrecision(precision=precision).use(BackwardIdentity)

        self.noise1_pre = GaussianNoise(leakage=leakage, precision=precision)
        self.noise1_post = GaussianNoise(leakage=leakage, precision=precision)
        self.noise2_pre = GaussianNoise(leakage=leakage, precision=precision)
        self.noise2_post = GaussianNoise(leakage=leakage, precision=precision)
        self.noise3_pre = GaussianNoise(leakage=leakage, precision=precision)
        self.noise3_post = GaussianNoise(leakage=leakage, precision=precision)

        self.activation1 = activation_class()
        self.activation2 = activation_class()
        self.activation3 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)
        activation_class.initialise_(self.linear3.weight)

        if approach == "full":
            pass
        if approach == "no_norm":
            self.norm1_pre.use(BackwardIdentity)
            self.norm1_post.use(BackwardIdentity)
            self.norm2_pre.use(BackwardIdentity)
            self.norm2_post.use(BackwardIdentity)
            self.norm3_pre.use(BackwardIdentity)
            self.norm3_post.use(BackwardIdentity)
        if approach == "use_norm_forward":
            self.norm1_pre.use(BackwardUsingForward)
            self.norm1_post.use(BackwardUsingForward)
            self.norm2_pre.use(BackwardUsingForward)
            self.norm2_post.use(BackwardUsingForward)
            self.norm3_pre.use(BackwardUsingForward)
            self.norm3_post.use(BackwardUsingForward)

        self.add_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.norm1_pre,
            self.rp1_pre,
            self.noise1_pre,
            self.linear1,
            self.noise1_post,
            self.norm1_post,
            self.rp1_post,
            self.activation1,

            self.norm2_pre,
            self.rp2_pre,
            self.noise2_pre,
            self.linear2,
            self.noise2_post,
            self.norm2_post,
            self.rp2_post,
            self.activation2,

            self.norm3_pre,
            self.rp3_pre,
            self.noise3_pre,
            self.linear3,
            self.noise3_post,
            self.norm3_post,
            self.rp3_post,
            self.activation3,
        )


class Linear4(BaseModel):
    def __init__(
            self, approach: str, in_features, out_features,
            activation_class: Type[Activation], norm_class: Type[Normalize], precision: int, leakage: float
    ):
        super(Linear4, self).__init__()
        self.activation_class = activation_class
        self.precision = precision

        self.linear1 = Linear(in_features=int(np.prod(in_features)), out_features=256).use(LinearBackpropagation)
        self.linear2 = Linear(in_features=self.linear1.out_features, out_features=128).use(LinearBackpropagation)
        self.linear3 = Linear(in_features=self.linear2.out_features, out_features=64).use(LinearBackpropagation)
        self.linear4 = Linear(in_features=self.linear3.out_features, out_features=out_features).use(
            LinearBackpropagation)

        self.norm1_pre = norm_class()
        self.norm1_post = norm_class()
        self.norm2_pre = norm_class()
        self.norm2_post = norm_class()
        self.norm3_pre = norm_class()
        self.norm3_post = norm_class()
        self.norm4_pre = norm_class()
        self.norm4_post = norm_class()

        self.rp1_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp1_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp2_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp3_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp3_post = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp4_pre = ReducePrecision(precision=precision).use(BackwardIdentity)
        self.rp4_post = ReducePrecision(precision=precision).use(BackwardIdentity)

        self.noise1_pre = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise1_post = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise2_pre = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise2_post = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise3_pre = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise3_post = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise4_pre = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)
        self.noise4_post = GaussianNoise(leakage=leakage, precision=precision).use(BackwardIdentity)

        self.activation1 = activation_class()
        self.activation2 = activation_class()
        self.activation3 = activation_class()
        self.activation4 = activation_class()

        activation_class.initialise_(self.linear1.weight)
        activation_class.initialise_(self.linear2.weight)
        activation_class.initialise_(self.linear3.weight)
        activation_class.initialise_(self.linear4.weight)

        if approach == "full":
            pass
        if approach == "no_norm":
            self.norm1_pre.use(BackwardIdentity)
            self.norm1_post.use(BackwardIdentity)
            self.norm2_pre.use(BackwardIdentity)
            self.norm2_post.use(BackwardIdentity)
            self.norm3_pre.use(BackwardIdentity)
            self.norm3_post.use(BackwardIdentity)
            self.norm4_pre.use(BackwardIdentity)
            self.norm4_post.use(BackwardIdentity)
        if approach == "use_norm_forward":
            self.norm1_pre.use(BackwardUsingForward)
            self.norm1_post.use(BackwardUsingForward)
            self.norm2_pre.use(BackwardUsingForward)
            self.norm2_post.use(BackwardUsingForward)
            self.norm3_pre.use(BackwardUsingForward)
            self.norm3_post.use(BackwardUsingForward)
            self.norm4_pre.use(BackwardUsingForward)
            self.norm4_post.use(BackwardUsingForward)
        if approach == "no_norm":
            self.norm1_pre.use(BackwardIdentity)
            self.norm1_post.use(BackwardIdentity)
            self.norm2_pre.use(BackwardIdentity)
            self.norm2_post.use(BackwardIdentity)
            self.norm3_pre.use(BackwardIdentity)
            self.norm3_post.use(BackwardIdentity)
            self.norm4_pre.use(BackwardIdentity)
            self.norm4_post.use(BackwardIdentity)
        if approach == "use_norm_forward":
            self.norm1_pre.use(BackwardUsingForward)
            self.norm1_post.use(BackwardUsingForward)
            self.norm2_pre.use(BackwardUsingForward)
            self.norm2_post.use(BackwardUsingForward)
            self.norm3_pre.use(BackwardUsingForward)
            self.norm3_post.use(BackwardUsingForward)
            self.norm4_pre.use(BackwardUsingForward)
            self.norm4_post.use(BackwardUsingForward)

        self.add_sequential_relation(
            Flatten(start_dim=1),
            self.backward.STOP,

            self.norm1_pre,
            self.rp1_pre,
            self.noise1_pre,
            self.linear1,
            self.noise1_post,
            self.norm1_post,
            self.rp1_post,
            self.activation1,

            self.norm2_pre,
            self.rp2_pre,
            self.noise2_pre,
            self.linear2,
            self.noise2_post,
            self.norm2_post,
            self.rp2_post,
            self.activation2,

            self.norm3_pre,
            self.rp3_pre,
            self.noise3_pre,
            self.linear3,
            self.noise3_post,
            self.norm3_post,
            self.rp3_post,
            self.activation3,

            self.norm4_pre,
            self.rp4_pre,
            self.noise4_pre,
            self.linear4,
            self.noise4_post,
            self.norm4_post,
            self.rp4_post,
            self.activation4,
        )


def main(
        name, data_folder,
        model_class: Type[Linear2],
        approach, activation_class, norm_class_y, norm_class_w, precision_y, precision_w, leakage_y, leakage_w,
        optimiser_class, optimiser_parameters,
        dataset, batch_size, epochs
):
    set_device("cpu")
    device, is_cuda = is_using_cuda()
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

    model = model_class(
        approach=approach,
        in_features=input_shape[1:],
        out_features=len(classes),
        activation_class=activation_class,
        norm_class=norm_class_y,
        precision=precision_y,
        leakage=leakage_y,
    )
    if TENSORBOARD:
        model.create_tensorboard(paths.tensorboard)

    model.compile(device=get_device(), layer_data=True)
    model.loss_fn = nn.CrossEntropyLoss()
    model.accuracy_fn = cross_entropy_loss_accuracy

    if precision_w is not None:
        StochasticReducePrecisionParameter.convert_model(
            model,
            precision=precision_w,
            use_zero_pseudo_tensor=False
        )

    model.optimizer = StochasticReducePrecisionOptimizer(
        optimiser_class,
        model.parameters(),
        **optimiser_parameters
    )

    parameter_log = {
        "model": model_class.__name__,
        "approach": approach,
        "dataset": dataset.__name__,
        "activation_class": model.activation_class.__name__,
        'norm_class_y': norm_class_y.__name__,
        'norm_class_w': norm_class_w.__name__,
        'precision_y': str(precision_y),
        'precision_w': str(precision_w),
        'leakage_y': str(leakage_y),
        'leakage_w': str(leakage_w),
        "loss_class": model.loss_fn.__class__.__name__,
        "model_class": model.__class__.__name__,
        'optimiser_class': model.optimizer.__class__.__name__,
        'batch_size': batch_size,
    }

    with open(log_file, "a+") as file:
        file.write(json.dumps(parameter_log, sort_keys=True, indent=2) + "\n\n")
        file.write(str(model.optimizer) + "\n\n")

        file.write(str(model) + "\n\n")
        file.write(summary(model, input_size=tuple(input_shape[1:])) + "\n\n")

    data = next(iter(train_loader))[0]
    data = data.to(model.device)
    save_graph(path_join(paths.logs, paths.name), model(data), model.named_parameters())

    apply_fn = [normalize_parameter(norm_class_w), add_gaussian_noise(leakage_w, precision_w)]
    for epoch in range(epochs):
        train_loss, train_accuracy = model.train_on(train_loader, epoch=epoch, apply_fn=apply_fn)
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
    print()
