"""Classes for measuring reconstruction times."""
from time import perf_counter
from typing import Union

import torch


class CPUTimer:
    """CPUTimer class."""

    def __enter__(self):
        """Enter CPUTimer class."""
        self.start_time = perf_counter()

    def __exit__(self, *args):
        """Exit CPUTimer class."""
        self.end_time = perf_counter()
        self.execution_time = self.end_time - self.start_time


class CUDATimer:
    """CUDATimer class."""

    def __enter__(self):
        """Enter CUDATimer class."""
        self.starter, self.ender = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)
        self.starter.record()

    def __exit__(self, *args):
        """Exit CUDATimer class."""
        self.ender.record()
        torch.cuda.synchronize()
        self.execution_time = self.starter.elapsed_time(self.ender)


def select_timer(device):
    """Select a timer based on device."""
    if device == 'cpu':
        timer = CPUTimer()

    elif device == 'cuda':
        timer = CUDATimer()
    return timer
