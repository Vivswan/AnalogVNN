from __future__ import annotations

from typing import Tuple

import torch

__all__ = ['CPUCuda', 'is_cpu_cuda']


class CPUCuda:
    """CPUCuda is a class that can be used to get, check and set the device.

    Attributes:
        _device (torch.device): The device.
        device_name (str): The name of the device.
    """
    _device: torch.device
    device_name: str

    def __init__(self):
        """Initialize the CPUCuda class."""
        self._device = None
        self.device_name = None
        self.reset_device()

    def reset_device(self):
        """Reset the device to the default device.

        Returns:
            CPUCuda: self
        """
        self.set_device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        return self

    def set_device(self, device_name: str) -> CPUCuda:
        """Set the device to the given device name.

        Args:
            device_name (str): the device name.

        Returns:
            CPUCuda: self
        """

        self._device = torch.device(device_name)
        self.device_name = self._device.type
        return self

    @property
    def is_cuda(self) -> bool:
        """Check if the device is cuda.

        Returns:
            bool: True if the device is cuda, False otherwise.
        """
        return "cuda" in self.device_name

    @property
    def device(self) -> torch.device:
        """Get the device.

        Returns:
            torch.device: the device.
        """
        return self._device

    @property
    def is_using_cuda(self) -> Tuple[torch.device, bool]:
        """Check if the device is cuda.

        Returns:
            tuple: the device and True if the device is cuda, False otherwise.
        """
        return self.device, self.is_cuda

    def get_module_device(self, module) -> torch.device:
        """Get the device of the given module.

        Args:
            module (torch.nn.Module): the module.

        Returns:
            torch.device: the device of the module.
        """

        # noinspection PyBroadException
        try:
            device: torch.device = getattr(module, "device", None)
            if device is None:
                device = next(module.parameters()).device
            return device
        except Exception:
            return self._device


is_cpu_cuda: CPUCuda = CPUCuda()
"""CPUCuda: The CPUCuda instance."""
