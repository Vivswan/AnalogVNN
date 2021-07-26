import torch

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

def set_device(device_name):
    global DEVICE_NAME
    DEVICE_NAME = device_name

def is_cuda() -> bool:
    return "cuda" in DEVICE_NAME

def get_device() -> (torch.device, bool):
    return torch.device(DEVICE_NAME)

def is_using_cuda() -> (torch.device, bool):
    return get_device(), is_cuda()
