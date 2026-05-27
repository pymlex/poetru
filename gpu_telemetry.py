from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GpuSnapshot:
    """Single timestamped snapshot of GPU load and memory."""

    utilisation_percent: float
    mem_used_bytes: int
    mem_total_bytes: int


class GpuTelemetry:
    """Thin NVML wrapper for utilisation and memory readouts."""

    def __init__(self, device_index: int = 0) -> None:
        import pynvml

        self._nv = pynvml
        self._idx = device_index
        self._nv.nvmlInit()
        self._handle = self._nv.nvmlDeviceGetHandleByIndex(device_index)

    def close(self) -> None:
        """Releases NVML resources."""

        self._nv.nvmlShutdown()

    def read(self) -> GpuSnapshot:
        """Samples utilisation and memory for the configured device."""

        util = float(self._nv.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        mem = self._nv.nvmlDeviceGetMemoryInfo(self._handle)
        return GpuSnapshot(
            utilisation_percent=util,
            mem_used_bytes=int(mem.used),
            mem_total_bytes=int(mem.total),
        )


def seed_everything(seed: int) -> None:
    """Fixes RNG seeds for NumPy and PyTorch."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
