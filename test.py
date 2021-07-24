import torch
from torch import nn, Tensor, autograd


def to_matrix(tensor: Tensor):
    if len(tensor.size()) == 1:
        tensor = tensor.reshape(tuple([1] + list(tensor.size())))
    return tensor


class LinearFunction(autograd.Function):
    backward_pass = ["backpropagation", "feedback_alignment", "direct_feedback_alignment"]

    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        y = x @ weight.t()
        if bias is not None:
            y += bias
        return y

    @staticmethod
    def backward_bp(ctx, grad_output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        LinearFunction.backward_bp(ctx, grad_output)
        print(f"grad_output: {grad_output}")
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_output = to_matrix(grad_output)
        x = to_matrix(x)
        weight = to_matrix(weight)
        bias = to_matrix(bias)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(x)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_output, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 1)

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)


if __name__ == '__main__':
    torch.manual_seed(0)

    model = nn.Sequential(
        Linear(input_features=1, output_features=1),
        nn.Dropout(),
        Linear(input_features=1, output_features=1)
    )
    data = torch.ones((1,))
    target = torch.ones((1,))

    # print(summary(model, (1, 1), include_self=True))

    # with torch.no_grad():
    #     for p in model.parameters():
    #         p.data = torch.zeros_like(p.data)
    #
    #     print(f"output before: {model(x)}")
    #     for p in model.named_parameters():
    #         print(f"p before  - {p[0]} ({p[1].requires_grad}): {p[1].data}")

    print()
    for i in range(5):
        model.train()

        output = model(data)
        # MSELoss
        loss = torch.mean((output - target) ** 2)

        print(f"loss ({i}): {loss.item()}")

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                # print(f"grad: {p.grad}")
                p.copy_(p - 0.1 * p.grad)

        # break
    #
    # print()
    # with torch.no_grad():
    #     for p in model.named_parameters():
    #         print(f"p after  - {p[0]} ({p[1].requires_grad}): {p[1].data}: {p[1].grad}")
    #
    #     print(f"output after : {model(data)}")

    import torch

    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    Q = 3 * a ** 3 - b ** 2
    external_grad = torch.tensor([1., 1.])
    Q.backward(gradient=external_grad)
    print(Q)
    print()
