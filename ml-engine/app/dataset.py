"""Load and split the public benchmark before provisioning hospital data."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    dataset = load_breast_cancer()
    features = dataset.data
    feature_names = tuple(str(name) for name in dataset.feature_names)
    if features.shape != (569, 30) or not np.isfinite(features).all():
        raise ValueError("Unexpected feature matrix")
    if len(feature_names) != 30 or len(set(feature_names)) != 30:
        raise ValueError("Unexpected feature schema")
    if (
        tuple(dataset.target_names) != ("malignant", "benign")
        or dataset.target.shape != (569,)
        or set(dataset.target) != {0, 1}
    ):
        raise ValueError("Unexpected label schema")
    return features, (1 - dataset.target).astype(np.int64), feature_names


def make_train_test_split(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(
        np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=seed
    )
