import time
from enum import Enum
from math import exp

import torch
from torch import nn, optim

from nn.BaseModel import BaseModel
from nn.activations.ReLU import LeakyReLU
from nn.layers.Linear import Linear
from nn.optimizer.ReducePrecisionOptimizer import PrecisionUpdateTypes, ReducePrecisionOptimizer
from nn.parameters.ReducePrecisionParameter import ReducePrecisionParameter
from nn.utils.is_using_cuda import get_device, set_device
from nn.utils.make_dot import make_dot


def save_graph(filename, output, named_parameters):
    make_dot(output, params=dict(named_parameters)).render(filename + ".svg", cleanup=True)


class TestModel(BaseModel):
    class BackPassTypes(Enum):
        default = "default"
        BP = "BP"
        FA = "FA"
        DFA = "DFA"
        RFA = "RFA"
        RDFA = "RDFA"

    def __init__(
            self,
            approach: 'TestModel.BackPassTypes',
            std,
            activation_class,
    ):
        super(TestModel, self).__init__()
        self.approach = approach
        self.std = std
        self.activation_class = activation_class

        self.linear1 = Linear(in_features=64, out_features=16)
        self.activation1 = activation_class()

        self.linear2 = Linear(in_features=16, out_features=8)
        self.activation2 = activation_class()

        self.linear3 = Linear(in_features=8, out_features=target.size()[1])
        self.activation3 = activation_class()

        activation_class.initialise_(None, self.linear1.weight)
        activation_class.initialise_(None, self.linear2.weight)
        activation_class.initialise_(None, self.linear3.weight)

        if approach == TestModel.BackPassTypes.default:
            self.backward.use_default_graph = True

        if approach == TestModel.BackPassTypes.BP:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.activation3,
                self.linear3.backpropagation,
                self.activation2,
                self.linear2.backpropagation,
                self.activation1,
                self.linear1.backpropagation
            )

        if approach == TestModel.BackPassTypes.FA or approach == TestModel.BackPassTypes.RFA:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.activation3,
                self.linear3.feedforward_alignment,
                self.activation2,
                self.linear2.feedforward_alignment,
                self.activation1,
                self.linear1.feedforward_alignment
            )

        if approach == TestModel.BackPassTypes.DFA or approach == TestModel.BackPassTypes.RDFA:
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear3.direct_feedforward_alignment.pre_backward,
                self.activation3,
                self.linear3.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear2.direct_feedforward_alignment.pre_backward,
                self.activation2,
                self.linear2.direct_feedforward_alignment
            )
            self.backward.add_relation(
                self.backward.OUTPUT,
                self.linear1.direct_feedforward_alignment.pre_backward,
                self.activation1,
                self.linear1.direct_feedforward_alignment
            )

        if approach == TestModel.BackPassTypes.RFA or approach == TestModel.BackPassTypes.RDFA:
            self.linear1.feedforward_alignment.is_fixed = False
            self.linear2.feedforward_alignment.is_fixed = False
            self.linear3.feedforward_alignment.is_fixed = False
            self.linear1.direct_feedforward_alignment.is_fixed = False
            self.linear2.direct_feedforward_alignment.is_fixed = False
            self.linear3.direct_feedforward_alignment.is_fixed = False

        if std is not None:
            self.linear1.feedforward_alignment.std = std
            self.linear2.feedforward_alignment.std = std
            self.linear3.feedforward_alignment.std = std
            self.linear1.direct_feedforward_alignment.std = std
            self.linear2.direct_feedforward_alignment.std = std
            self.linear3.direct_feedforward_alignment.std = std

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        return x


def accuracy_fn(output, target):
    return exp(-float(
        torch.sum(torch.abs(output - target))
    )) * 100


def main(name, parameter_precision, model_parameters, weight_update_type):
    torch.manual_seed(seed)
    model = TestModel(**model_parameters)
    if TENSORBOARD:
        model.create_tensorboard(f"{DATA_FOLDER}/{name}")
    model.compile(device=get_device(), layer_data=False)

    model.loss_fn = nn.MSELoss()
    model.accuracy_fn = accuracy_fn
    if parameter_precision is not None:
        ReducePrecisionParameter.convert_model(
            model,
            precision=parameter_precision,
            use_zero_pseudo_tensor=False
        )

    model.optimizer = ReducePrecisionOptimizer(
        optim.Adam,
        model.parameters(),
        weight_update_type=weight_update_type,
        lr=0.03
    )
    # print("optimizer: ", model.optimizer)
    # model.optimizer = optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(epochs):
        loss, accuracy = model.train_on([(data, target)], epoch=epoch)

        if epoch % int(epochs / 5) == 0:
            out = str(['%.3f' % member for member in model.output(data).tolist()[0]]).replace('\'', '')
            print(f"{name.rjust(7)}: {accuracy:.4f}% \t- loss: {loss:.4f} - output: {out} ({epoch})")

    if bool(torch.any(torch.isnan(model.output(data)))):
        raise ValueError

    if TENSORBOARD:
        model.tensorboard.tensorboard.add_hparams(
            hparam_dict={
                "approach": model.approach.value,
                "std": str(model.std),
                "activation_class": model.activation_class.__name__,
                "parameter_precision": str(parameter_precision),
                "weight_update_type": str(weight_update_type),
            },
            metric_dict={
                "loss": loss,
                "accuracy": accuracy,
                "accuracy_positive": accuracy_fn(torch.clamp(model.output(data), min=0), torch.clamp(target, min=0)),
                "accuracy_negative": accuracy_fn(torch.clamp(model.output(data), max=0), torch.clamp(target, max=0)),
            }
        )
    save_graph(f"{DATA_FOLDER}/{name}", model(data), model.named_parameters())
    print()


if __name__ == '__main__':
    DATA_FOLDER = f"C:/_data/tensorboard_SRP/"
    set_device("cpu")
    # if os.path.exists(DATA_FOLDER):
    #     shutil.rmtree(DATA_FOLDER)
    # os.mkdir(DATA_FOLDER)

    TENSORBOARD = True
    timestamp = str(int(time.time()))
    seed = int(time.time())
    # seed = 10

    data = (torch.rand((1, 64), device=get_device()) * 2) - 1
    data /= data.norm()
    target = torch.Tensor([[-1, -0.50, 0.50, 1]]).to(get_device())  # >= 2
    # target = torch.linspace(-1, 1, 9, device=get_device())  # >= 4
    # target = target.reshape((1, target.size()[0]))
    out = str(['%.3f' % member for member in target.tolist()[0]]).replace('\'', '')
    print(f"{'target'.rjust(43)}: {out}")

    epochs = 10000
    for approach in TestModel.BackPassTypes:
        model_parameters = {
            'approach': approach,
            'std': None,
            'activation_class': LeakyReLU
        }
        for weight_update_type in PrecisionUpdateTypes:
            for precision in [2, 4, 8, 16, 32, 64]:
                main(f"{timestamp}-RP-{model_parameters['approach'].value}-{precision}-{weight_update_type.value}",
                     precision, model_parameters, weight_update_type)
        # main(f"{timestamp}-RP-max", None, model_parameters)
