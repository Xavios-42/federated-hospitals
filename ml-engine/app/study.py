"""Run or replot the multi-seed benchmark study."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .algorithms import ALGORITHMS, Strategy
from .coordinator import train_federated
from .dataset import load_dataset, make_train_test_split
from .metrics import evaluate_weights, summarize_validation
from .models import create_model
from .partitioning import PARTITION_MODES
from .provisioning import build_clients, split_hospitals
from .training import LocalTrainingConfig, fit_model, log_features

BASELINE = "centralized_logistic"
METRIC_NAMES = ("loss", "accuracy", "precision", "recall", "f1")


def _validate_config(config: dict) -> tuple[LocalTrainingConfig, list[Strategy]]:
    if not isinstance(config, dict) or set(config) != {
        "test_seed", "seeds", "rounds", "training", "strategies"
    }:
        raise ValueError("Expected test_seed, seeds, rounds, training, and strategies")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in [config["test_seed"], *seeds]):
        raise ValueError("All seeds must be integers from 0 through 4294967295")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Study seeds must be distinct")
    if type(config["rounds"]) is not int or config["rounds"] < 1:
        raise ValueError("rounds must be a positive integer")
    training = config["training"]
    if not isinstance(training, dict) or set(training) != {"epochs", "batch_size", "learning_rate"}:
        raise ValueError("training requires epochs, batch_size, and learning_rate")
    strategies = config["strategies"]
    if not isinstance(strategies, list) or any(
        not isinstance(item, dict)
        or "algorithm" not in item
        or item.keys() - {"algorithm", "mu", "beta", "server_lr"}
        for item in strategies
    ):
        raise ValueError("Each strategy requires algorithm with optional mu, beta, and server_lr")
    parsed = [Strategy(**item) for item in strategies]
    if sorted(item.algorithm for item in parsed) != sorted(ALGORITHMS):
        raise ValueError("The study requires exactly one of each federated strategy")
    return LocalTrainingConfig(**training), parsed


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    malignant = int(labels.sum())
    return {"samples": len(labels), "benign": len(labels) - malignant, "malignant": malignant}


def _centralized_baseline(
    train: TensorDataset,
    validation: tuple[torch.Tensor, torch.Tensor],
    config: LocalTrainingConfig,
    seed: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    # The baseline is a separate benchmark over the public data.
    model = fit_model(train, create_model(seed).state_dict(), config, seed)
    weights = model.state_dict()
    validation_summary = evaluate_weights(weights, *validation)
    return summarize_validation([validation_summary]), weights


def summarize_runs(runs: list[dict]) -> list[dict]:
    summary = []
    for mode in PARTITION_MODES:
        for algorithm in (*ALGORITHMS, BASELINE):
            group = [run for run in runs if run["mode"] == mode and run["algorithm"] == algorithm]
            metrics = {}
            for name in METRIC_NAMES:
                values = [run["test"][name] for run in group]
                metrics[name] = {
                    "mean": mean(values),
                    "sd": stdev(values) if len(values) > 1 else 0.0,
                }
            summary.append({"mode": mode, "algorithm": algorithm, "seeds": len(group), "test": metrics})
    return summary


def run_study(config: dict) -> dict:
    training, strategies = _validate_config(config)
    features, labels, names = load_dataset()
    development, test = make_train_test_split(labels, config["test_seed"])
    test_features = log_features(features[test])
    test_labels = torch.tensor(labels[test], dtype=torch.float32)
    baseline_training = replace(training, epochs=training.epochs * config["rounds"])
    runs = []
    partitions = []
    for mode in PARTITION_MODES:
        for seed in config["seeds"]:
            splits = split_hospitals(development, labels, mode, seed)
            clients = build_clients(features, labels, splits, shared_log_transform=True)
            partition_hash = hashlib.sha256()
            for train, validation in splits.values():
                for indices in (train, validation):
                    partition_hash.update(len(indices).to_bytes(4, "little"))
                    partition_hash.update(indices.astype("<i8").tobytes())
            partitions.append({
                "mode": mode,
                "seed": seed,
                "sha256": partition_hash.hexdigest(),
                "hospitals": {
                    site: {"train": _class_counts(labels[train]), "validation": _class_counts(labels[validation])}
                    for site, (train, validation) in splits.items()
                },
            })
            for strategy in strategies:
                report, weights = train_federated(clients, training, strategy, config["rounds"], seed)
                runs.append({
                    "mode": mode,
                    "seed": seed,
                    "algorithm": strategy.algorithm,
                    "strategy": asdict(strategy),
                    "train_samples": sum(report["hospital_training_samples"].values()),
                    "validation": report["history"][-1]["validation"],
                    "test": summarize_validation([evaluate_weights(weights, test_features, test_labels)]),
                    **report,
                })
            train = np.concatenate([train for train, _ in splits.values()])
            validation = np.concatenate([validation for _, validation in splits.values()])
            baseline_validation, weights = _centralized_baseline(
                TensorDataset(log_features(features[train]), torch.tensor(labels[train], dtype=torch.float32)),
                (log_features(features[validation]), torch.tensor(labels[validation], dtype=torch.float32)),
                baseline_training,
                seed,
            )
            runs.append({
                "mode": mode,
                "seed": seed,
                "algorithm": BASELINE,
                "train_samples": len(train),
                "completed_rounds": 0,
                "communication": {
                    "initial_download_model_bytes": 0,
                    "upload_model_bytes": 0,
                    "download_model_bytes": 0,
                    "total_model_bytes": 0,
                },
                "validation": baseline_validation,
                "test": summarize_validation([evaluate_weights(weights, test_features, test_labels)]),
            })
    return {
        "config": config,
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "scikit-learn", "torch", "matplotlib")},
        },
        "model": "logistic",
        "preprocessing": "log1p",
        "positive_class": "malignant",
        "feature_names": names,
        "development": _class_counts(labels[development]),
        "test": {
            **_class_counts(labels[test]),
            "sha256": hashlib.sha256(np.sort(test).astype("<i8").tobytes()).hexdigest(),
        },
        "baseline_training": asdict(baseline_training),
        "partitions": partitions,
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def main() -> None:
    from .study_output import render_plots, write_reports

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train all models and write reports")
    run.add_argument("--config", type=Path, required=True)
    plot = commands.add_parser("plot", help="render plots from an existing study JSON")
    plot.add_argument("--report", type=Path, required=True)
    for command in (run, plot):
        command.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError("Output directory already exists")
        if args.command == "run":
            torch.set_num_threads(1)
            report = run_study(json.loads(args.config.read_text(encoding="utf-8")))
        else:
            report = json.loads(args.report.read_text(encoding="utf-8"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / "report"
            staging.mkdir()
            if args.command == "run":
                write_reports(report, staging)
            render_plots(report, staging)
            staging.rename(output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote study outputs to {output}")


if __name__ == "__main__":
    main()
