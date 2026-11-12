"""Run standalone or federated training at the three simulated hospitals."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .algorithms import Strategy
from .clients import LocalTrainingConfig
from .coordinator import run_rounds
from .models import create_model
from .partitioning import PARTITION_MODES
from .provisioning import provision_hospitals


def run(mode: str, seed: int, training: LocalTrainingConfig) -> dict:
    clients, development_samples, test_samples = provision_hospitals(mode, seed)
    initial_state = create_model(seed).state_dict()
    hospitals = {}
    for offset, (hospital_id, client) in enumerate(clients.items()):
        local_seed = (seed + offset) % 2**32
        update = client.train_local(initial_state, training, round_index=0, seed=local_seed)
        metrics = update.summary_metrics
        hospitals[hospital_id] = {
            "train": {
                "samples": update.num_train_samples,
                "benign": update.num_train_samples - metrics["train_malignant"],
                "malignant": metrics["train_malignant"],
            },
            "validation": {
                "samples": metrics["validation_samples"],
                "benign": metrics["validation_samples"] - metrics["validation_malignant"],
                "malignant": metrics["validation_malignant"],
                "loss": metrics["validation_loss"],
                "accuracy": metrics["validation_correct"] / metrics["validation_samples"],
            },
        }
    return {
        "mode": mode,
        "seed": seed,
        "model": "logistic",
        "training": asdict(training),
        "development_samples": development_samples,
        "reserved_test_samples": test_samples,
        "hospitals": hospitals,
    }


def run_federated(
    mode: str, seed: int, training: LocalTrainingConfig, settings: dict
) -> dict:
    required = {"algorithm", "rounds"}
    allowed = required | {"mu", "beta", "server_lr"}
    if not isinstance(settings, dict) or not required <= settings.keys() <= allowed:
        raise ValueError("federated settings require algorithm and rounds, with optional mu, beta, server_lr")
    strategy = Strategy(**{key: value for key, value in settings.items() if key != "rounds"})
    clients, development_samples, test_samples = provision_hospitals(
        mode, seed, shared_log_transform=True
    )
    report = run_rounds(clients, training, strategy, settings["rounds"], seed)
    return {
        "mode": mode,
        "seed": seed,
        "model": "logistic",
        "preprocessing": "log1p",
        "positive_class": "malignant",
        "training": asdict(training),
        "strategy": asdict(strategy),
        "development_samples": development_samples,
        "reserved_test_samples": test_samples,
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON run settings")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        required = {"mode", "seed", "epochs", "batch_size", "learning_rate"}
        if not isinstance(settings, dict) or not required <= settings.keys() <= required | {"federated"}:
            raise ValueError("Expected mode, seed, epochs, batch_size, and learning_rate")
        mode = settings["mode"]
        if mode not in PARTITION_MODES:
            raise ValueError("mode must be iid or non-iid")
        seed = settings["seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer from 0 through 4294967295")
        training = LocalTrainingConfig(
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
        )
        if "federated" in settings:
            report = run_federated(mode, seed, training, settings["federated"])
        else:
            report = run(mode, seed, training)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
