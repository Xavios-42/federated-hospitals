"""Private stdin/stdout protocol used by the Spring Boot experiment worker."""

import argparse
import json
import sys
from importlib.util import find_spec


def emit(event_type: str, metrics: dict) -> None:
    print(json.dumps({"type": event_type, "metrics": metrics}, allow_nan=False), flush=True)


def run(config: dict) -> None:
    import torch

    from .algorithms import Strategy
    from .coordinator import train_federated
    from .dataset import load_dataset, make_train_test_split
    from .metrics import evaluate_weights, summarize_validation
    from .provisioning import build_clients, split_hospitals
    from .training import LocalTrainingConfig, log_features

    torch.set_num_threads(1)
    seed = config["seed"]
    training = LocalTrainingConfig(config["epochs"], config["batchSize"], config["learningRate"])
    strategy = Strategy(config["algorithm"], config["mu"], config["beta"], config["serverLearningRate"])
    features, labels, _ = load_dataset()
    development, test = make_train_test_split(labels, seed)
    splits = split_hospitals(development, labels, config["mode"], seed)
    clients = build_clients(features, labels, splits, shared_log_transform=True)
    report, weights = train_federated(
        clients, training, strategy, config["rounds"], seed,
        on_round=lambda metrics: emit("round", metrics),
    )
    test_summary = evaluate_weights(
        weights, log_features(features[test]), torch.tensor(labels[test], dtype=torch.float32)
    )
    emit("complete", {
        "test": summarize_validation([test_summary]),
        "communication": report["communication"],
        "hospital_training_samples": report["hospital_training_samples"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    if args.health:
        sys.exit(0 if all(find_spec(name) is not None for name in ("torch", "sklearn", "numpy")) else 1)
    run(json.load(sys.stdin))


if __name__ == "__main__":
    main()
