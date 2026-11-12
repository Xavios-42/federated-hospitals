"""Sample-weighted model aggregation and server momentum."""

from dataclasses import dataclass
from math import isfinite

import torch

from .clients import ClientUpdate
from .partitioning import HOSPITAL_IDS

ModelState = dict[str, torch.Tensor]
ALGORITHMS = ("fedavg", "fedprox", "fedavgm")


@dataclass(frozen=True)
class Strategy:
    algorithm: str
    mu: float = 0.0
    beta: float = 0.0
    server_lr: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError("algorithm must be fedavg, fedprox, or fedavgm")
        for name in ("mu", "beta", "server_lr"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.mu < 0 or not 0 <= self.beta < 1 or self.server_lr <= 0:
            raise ValueError("Expected mu >= 0, 0 <= beta < 1, and server_lr > 0")
        if self.algorithm != "fedprox" and self.mu != 0:
            raise ValueError("Only fedprox accepts nonzero mu")
        if self.algorithm != "fedavgm" and (self.beta != 0 or self.server_lr != 1):
            raise ValueError("Only fedavgm accepts custom beta and server_lr")

    @torch.no_grad()
    def aggregate(
        self,
        global_state: ModelState,
        updates: list[ClientUpdate],
        round_index: int,
        velocity: ModelState | None = None,
    ) -> tuple[ModelState, ModelState | None]:
        """Validate a full round and return its new weights and optional velocity."""
        if sorted(update.hospital_id for update in updates) != list(HOSPITAL_IDS):
            raise ValueError("Expected exactly one update from each of A, B, and C")
        updates = sorted(updates, key=lambda update: update.hospital_id)
        for update in updates:
            if type(update.round_index) is not int or update.round_index != round_index:
                raise ValueError(f"Hospital {update.hospital_id} returned the wrong round")
            if type(update.num_train_samples) is not int or update.num_train_samples <= 0:
                raise ValueError(f"Hospital {update.hospital_id} returned an invalid sample count")
            if update.model_state.keys() != global_state.keys():
                raise ValueError(f"Hospital {update.hospital_id} returned incompatible tensor keys")
            for name, reference in global_state.items():
                value = update.model_state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or value.device != reference.device
                    or value.layout != torch.strided
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Hospital {update.hospital_id} returned invalid tensor {name}")

        total = sum(update.num_train_samples for update in updates)
        use_momentum = self.algorithm == "fedavgm"
        averaged = {name: torch.zeros_like(weight) for name, weight in global_state.items()}
        for update in updates:
            fraction = update.num_train_samples / total
            for name, accumulator in averaged.items():
                value = update.model_state[name]
                if use_momentum:
                    value = value - global_state[name]
                accumulator.add_(value, alpha=fraction)

        if use_momentum:
            if velocity is not None:
                for name, value in averaged.items():
                    value.add_(velocity[name], alpha=self.beta)
            state = {
                name: value + self.server_lr * averaged[name]
                for name, value in global_state.items()
            }
            next_velocity = averaged
        else:
            state, next_velocity = averaged, None
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Aggregation produced non-finite weights")
        return state, next_velocity
