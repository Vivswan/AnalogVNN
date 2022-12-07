from torch import Tensor, nn

from nn.utils.is_cpu_cuda import is_cpu_cuda
from nn.utils.make_dot import make_dot


def save_graph(filename, module: nn.Module, data: Tensor):
    data: Tensor = data.to(is_cpu_cuda.get_module_device(module))
    data: Tensor = data.clone()
    data.detach_()
    data.requires_grad_(True)

    training_status = module.training
    module.eval()

    output = module(data)

    named_parameters = dict(module.named_parameters())
    named_parameters["input"] = data
    named_parameters["output"] = output

    make_dot(output, params=named_parameters).render(filename, format="svg", cleanup=True)

    module.train(training_status)
