import torch
from torch import nn

from nn.Sequential import Sequential
from nn.layers.base_layer import BaseLayer, BackwardPass
from nn.utils.summary import summary


class Linear(BaseLayer):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 1)

    def forward(self, x):
        y = x @ self.weight.t()
        if self.bias is not None:
            y += self.bias
        return y


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'



if __name__ == '__main__':
    layer_1 = Linear(in_features=1, out_features=1)
    layer_2 = Linear(in_features=1, out_features=1)
    layer_3 = Linear(in_features=1, out_features=1)


    model = Sequential(
        layer_1,
        layer_2,
        layer_3,
    )

    print(model)
    print(summary(model, input_size=(1,)))
    layer_1.backward_pass_from(layer_3, BackwardPass.DFA)


    # data = torch.ones((1,))
    # target = torch.ones((1,))
    #
    # for i in range(5):
    #     model.train()
    #
    #     output = model(data)
    #     # MSELoss
    #     loss = torch.mean((output - target) ** 2)
    #
    #     print(f"loss ({i}): {loss.item()}")
    #
    #     model.zero_grad()
    #     loss.backward()
    #
    #     with torch.no_grad():
    #         for p in model.parameters():
    #             # print(f"grad: {p.grad}")
    #             p.copy_(p - 0.1 * p.grad)
    #
    #     # break
