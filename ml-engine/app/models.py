"""Shared binary classifier."""

import torch
from torch import nn


def create_model(seed: int) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Linear(30, 1, device="cpu", dtype=torch.float32)
