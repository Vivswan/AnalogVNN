from enum import Enum

from torch import nn

class BackwardPass(Enum):
    BACKPROPAGATION = "backpropagation"
    BP = "backpropagation"

    FEEDBACK_ALIGNMENT = "feedback_alignment"
    FA = "feedback_alignment"

    DIRECT_FEEDBACK_ALIGNMENT = "direct_feedback_alignment"
    DFA = "direct_feedback_alignment"

class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()

    def backward_pass_from(self, layer: nn.Module, pass_type: BackwardPass = BackwardPass.BP):
        pass

    def forward(self, x):
        pass

    def backward_backpropagation(self, grad_output):
        pass

    def backward_feedback_alignment(self, grad_output):
        pass

    def backward_direct_feedback_alignment(self, grad_output):
        pass

    def backward(self):
        pass