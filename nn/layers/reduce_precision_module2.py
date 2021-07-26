import torch
from torch import nn, autograd, Tensor

from nn.layers.reduce_precision_layer import ReducePrecision
from nn.utils.summary import summary


class ReducePrecisionModuleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, module: nn.Module, reduce_precision: ReducePrecision):
        ctx.save_for_backward(module, reduce_precision)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ReducePrecisionModule(nn.Module):
    def __init__(self, module: nn.Module, reduce_precision: ReducePrecision) -> None:
        super(ReducePrecisionModule, self).__init__()
        self.reduce_precision = reduce_precision
        self.module = module

    def forward(self, x: Tensor):
        return ReducePrecisionModuleFunction.apply(x, self.module, self.reduce_precision)

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.Tensor([1])
    y = torch.Tensor([1])

    linear = nn.Linear(1, 1)
    reduce_precision = ReducePrecision(precision=8)
    model = ReducePrecisionModule(module=linear, reduce_precision=reduce_precision)

    print(summary(model, input_size=(1, 1)))

    # noinspection DuplicatedCode
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.zeros_like(p.data)

        print(f"output before: {model(x)}")
        for p in model.named_parameters():
            print(f"p before  - {p[0]} ({p[1].requires_grad}): {p[1].data}")
