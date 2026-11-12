"""Seeded assignment of development indices to the simulated hospitals."""

import numpy as np
from sklearn.model_selection import train_test_split

HOSPITAL_IDS = ("A", "B", "C")
PARTITION_MODES = ("iid", "non-iid")


def partition_hospitals(
    train_indices: np.ndarray, labels: np.ndarray, mode: str, seed: int
) -> dict[str, np.ndarray]:
    if mode not in PARTITION_MODES:
        raise ValueError("Unsupported partition mode")

    rng = np.random.default_rng(seed)
    class_chunks = []
    for label in (0, 1):
        shuffled = rng.permutation(train_indices[labels[train_indices] == label])
        if mode == "iid":
            chunks = np.array_split(shuffled, len(HOSPITAL_IDS))
            # Rotate remainder allocation for malignant rows to balance site sizes.
            if label == 1:
                chunks = chunks[-1:] + chunks[:-1]
        else:
            # Cumulative shares assigned to A and A+B, respectively.
            shares = (0.20, 0.53) if label == 0 else (0.60, 0.90)
            cutoffs = [round(len(shuffled) * share) for share in shares]
            chunks = np.split(shuffled, cutoffs)
        class_chunks.append(chunks)

    return {
        hospital_id: rng.permutation(np.concatenate(chunks))
        for hospital_id, chunks in zip(HOSPITAL_IDS, zip(*class_chunks))
    }


def split_hospital(
    indices: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(
        indices, test_size=0.20, stratify=labels[indices], random_state=seed
    )
