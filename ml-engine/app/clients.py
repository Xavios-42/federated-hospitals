"""Hospital-local data ownership, preprocessing, training, and validation."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset

from .metrics import evaluate_weights
from .partitioning import HOSPITAL_IDS
from .training import LocalTrainingConfig, fit_model, log_features


@dataclass(frozen=True)
class ClientUpdate:
    hospital_id: str
    round_index: int
    num_train_samples: int
    model_state: dict[str, torch.Tensor]
    summary_metrics: dict[str, float | int]


class HospitalClient:
    def __init__(
        self,
        hospital_id: str,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
        *,
        shared_log_transform: bool = False,
    ) -> None:
        if hospital_id not in HOSPITAL_IDS:
            raise ValueError("Unknown hospital ID")
        self.hospital_id = hospital_id
        self._scaler = None if shared_log_transform else StandardScaler().fit(train_features)
        self._train = TensorDataset(
            self._transform(train_features), torch.tensor(train_labels, dtype=torch.float32)
        )
        self._validation = (
            self._transform(validation_features),
            torch.tensor(validation_labels, dtype=torch.float32),
        )

    def _transform(self, features: np.ndarray) -> torch.Tensor:
        if self._scaler is None:
            return log_features(features)
        return torch.tensor(self._scaler.transform(features), dtype=torch.float32)

    def train_local(
        self,
        global_weights: dict[str, torch.Tensor],
        config: LocalTrainingConfig,
        *,
        round_index: int,
        seed: int,
        mu: float = 0.0,
    ) -> ClientUpdate:
        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        if not isfinite(mu) or mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        model = fit_model(self._train, global_weights, config, seed, mu)
        criterion = nn.BCEWithLogitsLoss()

        model.eval()
        with torch.no_grad():
            features, labels = self._validation
            logits = model(features).squeeze(-1)
            validation_loss = criterion(logits, labels).item()
            validation_correct = int(((logits >= 0) == labels.bool()).sum().item())

        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if not isfinite(validation_loss) or any(
            not torch.isfinite(value).all() for value in state.values()
        ):
            raise ValueError(f"Hospital {self.hospital_id} produced a non-finite update")

        return ClientUpdate(
            hospital_id=self.hospital_id,
            round_index=round_index,
            num_train_samples=len(self._train),
            model_state=state,
            summary_metrics={
                "validation_loss": validation_loss,
                "validation_correct": validation_correct,
                "validation_samples": len(labels),
                "train_malignant": int(self._train.tensors[1].sum().item()),
                "validation_malignant": int(labels.sum().item()),
            },
        )

    def validate_global(self, weights: dict[str, torch.Tensor]) -> dict[str, float | int]:
        """Evaluate broadcast weights locally and return aggregate statistics."""
        return evaluate_weights(weights, *self._validation)
