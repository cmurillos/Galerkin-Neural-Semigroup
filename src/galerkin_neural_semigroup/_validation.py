"""Small validation helpers shared by the public objects."""

from math import isfinite
from numbers import Integral, Real

import torch


def positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        qualifier = "nonnegative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return result


def hidden_widths(hidden):
    if isinstance(hidden, (str, bytes)) or not isinstance(hidden, (list, tuple)):
        raise TypeError("hidden must be a list or tuple of positive widths.")
    return tuple(positive_integer(width, "hidden width") for width in hidden)


def floating_dtype(dtype):
    if dtype is None:
        return torch.float64
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    return dtype


def compute_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return result
