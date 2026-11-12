"""Set up the in-process hospital simulation using the public benchmark."""

import numpy as np

from .clients import HospitalClient
from .dataset import load_dataset, make_train_test_split
from .partitioning import partition_hospitals, split_hospital


def provision_hospitals(
    mode: str, seed: int, *, shared_log_transform: bool = False
) -> tuple[dict[str, HospitalClient], int, int]:
    features, labels, _ = load_dataset()
    development, test = make_train_test_split(labels, seed)
    splits = split_hospitals(development, labels, mode, seed)
    clients = build_clients(features, labels, splits, shared_log_transform=shared_log_transform)
    return clients, len(development), len(test)


def split_hospitals(
    development: np.ndarray, labels: np.ndarray, mode: str, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    partitions = partition_hospitals(development, labels, mode, seed)
    return {
        hospital_id: split_hospital(indices, labels, (seed + offset) % 2**32)
        for offset, (hospital_id, indices) in enumerate(partitions.items())
    }


def build_clients(
    features: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    shared_log_transform: bool,
) -> dict[str, HospitalClient]:
    return {
        hospital_id: HospitalClient(
            hospital_id,
            features[train],
            labels[train],
            features[validation],
            labels[validation],
            shared_log_transform=shared_log_transform,
        )
        for hospital_id, (train, validation) in splits.items()
    }
