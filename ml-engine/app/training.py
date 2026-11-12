"""Training and preprocessing shared by clients and the public-data baseline."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import create_model


@dataclass(frozen=True)
class LocalTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        rate = self.learning_rate
        if type(rate) not in (int, float) or not isfinite(rate) or rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def log_features(features: np.ndarray) -> torch.Tensor:
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("Shared log1p transform requires finite, nonnegative features")
    return torch.tensor(np.log1p(features), dtype=torch.float32)


def fit_model(
    data: TensorDataset,
    initial_state: dict[str, torch.Tensor],
    config: LocalTrainingConfig,
    seed: int,
    mu: float = 0.0,
) -> nn.Linear:
    model = create_model(seed)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    batches = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for _ in range(config.epochs):
        for features, labels in batches:
            optimizer.zero_grad()
            loss = criterion(model(features).squeeze(-1), labels)
            if mu:
                squared_distance = sum(
                    (parameter - initial_state[name].detach()).square().sum()
                    for name, parameter in model.named_parameters()
                )
                loss = loss + (mu / 2) * squared_distance
            loss.backward()
            optimizer.step()
    return model
