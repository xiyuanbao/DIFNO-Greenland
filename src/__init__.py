"""DIFNO training code: model classes, losses, gradient surgery and helpers
factored out of the four training notebooks so the notebooks only keep configs,
data loading and the training loops.
"""

from .adam import Adam
from .device import get_device, move_model
from .grad_surgery import (
    _cagrad_two_task,
    _match_param_dtype,
    _mgda_two_task,
    pcgrad_two_task,
)
from .losses import AmplitudeWeightedLoss
from .models import (
    FNO_dse,
    SpectralConv2d_dse,
    VFT,
    VFT_changing_points,
    set_default_device,
)
from .utilities import count_params, percentage_difference, print_params_by_module

__all__ = [
    "Adam",
    "AmplitudeWeightedLoss",
    "FNO_dse",
    "SpectralConv2d_dse",
    "VFT",
    "VFT_changing_points",
    "_cagrad_two_task",
    "_match_param_dtype",
    "_mgda_two_task",
    "count_params",
    "get_device",
    "move_model",
    "pcgrad_two_task",
    "percentage_difference",
    "print_params_by_module",
    "set_default_device",
]
