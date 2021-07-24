from torch import nn
from torch.autograd import Function


class BaseFunction(Function):
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
    def feedback_alignment(ctx, grad_output):
        pass

    @staticmethod
    def direct_feedback_alignment(ctx, grad_output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class BaseLayer(nn.Module):
    pass
