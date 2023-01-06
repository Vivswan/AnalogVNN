import torch

__all__ = ['is_cpu_cuda']


class CPUCuda:
    def __init__(self):
        self.device: torch.device = None
        self.device_name: str = None
        self.reset_device()

    def set_device(self, device_name):
        self.device = torch.device(device_name)
        self.device_name = self.device.type

    def reset_device(self):
        self.set_device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    def is_cuda(self) -> bool:
        return "cuda" in self.device_name

    def get_device(self) -> (torch.device, bool):
        return self.device

    def is_using_cuda(self) -> (torch.device, bool):
        return self.get_device(), self.is_cuda()

    def get_module_device(self, module):
        try:
            device = getattr(module, "device", None)
            if device is None:
                device = next(module.parameters()).device
            return device
        except Exception:
            return self.device


is_cpu_cuda = CPUCuda()
