import torch


def is_using_cuda(device_name=None) -> (torch.device, bool):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = "cuda" in device_name
    device = torch.device(device_name)
    return device, use_cuda
