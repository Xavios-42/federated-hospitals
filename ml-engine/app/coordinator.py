"""Run full-participation rounds over the hospital client interface."""

from io import BytesIO
from collections.abc import Callable

import torch

from .algorithms import ModelState, Strategy
from .clients import HospitalClient, LocalTrainingConfig
from .metrics import summarize_validation
from .models import create_model
from .partitioning import HOSPITAL_IDS


def serialized_model_bytes(state: ModelState) -> int:
    with BytesIO() as buffer:
        torch.save(state, buffer)
        return buffer.tell()


def run_rounds(
    clients: dict[str, HospitalClient],
    training: LocalTrainingConfig,
    strategy: Strategy,
    rounds: int,
    seed: int,
) -> dict:
    report, _ = train_federated(clients, training, strategy, rounds, seed)
    return report


def train_federated(
    clients: dict[str, HospitalClient],
    training: LocalTrainingConfig,
    strategy: Strategy,
    rounds: int,
    seed: int,
    *,
    on_round: Callable[[dict], None] | None = None,
) -> tuple[dict, ModelState]:
    if set(clients) != set(HOSPITAL_IDS) or any(
        hospital_id != client.hospital_id for hospital_id, client in clients.items()
    ):
        raise ValueError("Expected one client each for A, B, and C")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")

    global_state = dict(create_model(seed).state_dict())
    velocity = None
    initial_bytes = len(clients) * serialized_model_bytes(global_state)
    history = []
    for round_index in range(1, rounds + 1):
        updates = [
            clients[hospital_id].train_local(
                global_state,
                training,
                round_index=round_index,
                seed=(seed + (round_index - 1) * len(clients) + offset) % 2**32,
                mu=strategy.mu,
            )
            for offset, hospital_id in enumerate(HOSPITAL_IDS)
        ]
        global_state, velocity = strategy.aggregate(global_state, updates, round_index, velocity)
        validation = [clients[hospital_id].validate_global(global_state) for hospital_id in HOSPITAL_IDS]
        history.append(
            {
                "round": round_index,
                "validation": summarize_validation(validation),
                "upload_model_bytes": sum(serialized_model_bytes(update.model_state) for update in updates),
                "download_model_bytes": len(clients) * serialized_model_bytes(global_state),
            }
        )
        if on_round is not None:
            on_round(history[-1])

    upload_bytes = sum(entry["upload_model_bytes"] for entry in history)
    download_bytes = initial_bytes + sum(entry["download_model_bytes"] for entry in history)
    report = {
        "completed_rounds": len(history),
        "hospital_training_samples": {
            update.hospital_id: update.num_train_samples for update in updates
        },
        "communication": {
            "initial_download_model_bytes": initial_bytes,
            "upload_model_bytes": upload_bytes,
            "download_model_bytes": download_bytes,
            "total_model_bytes": upload_bytes + download_bytes,
        },
        "history": history,
    }
    return report, global_state
