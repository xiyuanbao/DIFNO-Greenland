"""Small helpers shared by every training notebook: parameter counts and the
relative-L1 error (``mean|truth - test| / mean|truth| x 100``) used as the
headline test metric. ``count_params`` / ``percentage_difference`` come from the
DSE repository's ``_Utilities/utilities.py``.
"""

from functools import reduce
import operator

import torch


def count_params(model):
    """
    Print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def percentage_difference(truth, test):
    """
    Compute relative errors
    """
    difference = torch.mean(torch.abs(truth - test))/torch.mean(torch.abs(truth)) * 100
    return difference.item()


def print_params_by_module(model):
    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        print(f"{name:40s} {count}")
        total += count
    print(f"\nTotal parameters: {total}")
