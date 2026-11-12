"""Export aggregate study results and render reproducible PNG figures."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .algorithms import ALGORITHMS
from .partitioning import PARTITION_MODES

matplotlib.use("Agg")
from matplotlib import pyplot as plt


STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedavgm": "FedAvgM",
    "centralized_logistic": "Centralized logistic",
}
COLORS = ("#2459a6", "#c95612", "#087f70")
MODE_LABELS = {"iid": "IID", "non-iid": "Non-IID"}


def _flatten_metrics(metrics: dict, prefix: str) -> dict:
    return {
        **{f"{prefix}_{key}": value for key, value in metrics.items() if key != "confusion_matrix"},
        **{f"{prefix}_{key}": value for key, value in metrics["confusion_matrix"].items()},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report: dict, output: Path) -> None:
    (output / "study.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    final_rows = []
    round_rows = []
    for run in report["runs"]:
        identity = {key: run[key] for key in ("mode", "seed", "algorithm")}
        final_rows.append({
            **identity,
            "train_samples": run["train_samples"],
            "completed_rounds": run["completed_rounds"],
            **run["communication"],
            **_flatten_metrics(run["validation"], "validation"),
            **_flatten_metrics(run["test"], "test"),
        })
        for row in run.get("history", []):
            round_rows.append({
                **identity,
                "round": row["round"],
                "upload_model_bytes": row["upload_model_bytes"],
                "download_model_bytes": row["download_model_bytes"],
                **_flatten_metrics(row["validation"], "validation"),
            })
    summary_rows = [
        {
            "mode": row["mode"],
            "algorithm": row["algorithm"],
            "seeds": row["seeds"],
            **{
                f"test_{metric}_{statistic}": value
                for metric, stats in row["test"].items()
                for statistic, value in stats.items()
            },
        }
        for row in report["summary"]
    ]
    _write_csv(output / "runs.csv", final_rows)
    _write_csv(output / "rounds.csv", round_rows)
    _write_csv(output / "summary.csv", summary_rows)


def render_plots(report: dict, output: Path) -> None:
    seed_count = len(report["config"]["seeds"])
    spread_label = f"Mean ± sample SD across {seed_count} run seeds" if seed_count > 1 else "One run seed"
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, layout="constrained")
    for row, mode in enumerate(PARTITION_MODES):
        for column, (metric, title) in enumerate(
            (("loss", "Validation loss"), ("accuracy", "Validation accuracy"), ("f1", "Validation F1"))
        ):
            ax = axes[row, column]
            for algorithm, color in zip(ALGORITHMS, COLORS):
                histories = [
                    run["history"] for run in report["runs"]
                    if run["mode"] == mode and run["algorithm"] == algorithm
                ]
                steps = [entry["round"] for entry in histories[0]]
                values = np.array([[entry["validation"][metric] for entry in history] for history in histories])
                average = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1) if seed_count > 1 else np.zeros_like(average)
                ax.plot(steps, average, label=STRATEGY_LABELS[algorithm], color=color, linewidth=2)
                ax.fill_between(steps, average - sd, average + sd, color=color, alpha=0.13)
            ax.set_title(title)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.25)
            if column == 0:
                ax.set_ylabel(MODE_LABELS[mode])
            if row == 1:
                ax.set_xlabel("Communication round")
            if metric != "loss":
                ax.set_ylim(0, 1.03)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(f"Global model learning curves\n{spread_label}", fontsize=14)
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, layout="constrained")
    rows = {(row["mode"], row["algorithm"]): row["test"] for row in report["summary"]}
    positions = np.arange(len(STRATEGY_LABELS))
    for ax, metric, title in zip(axes, ("accuracy", "f1"), ("Test accuracy", "Test F1")):
        for mode, offset, color, marker in zip(PARTITION_MODES, (-0.12, 0.12), COLORS, ("o", "s")):
            stats = [rows[mode, algorithm][metric] for algorithm in STRATEGY_LABELS]
            ax.errorbar(
                [entry["mean"] for entry in stats], positions + offset,
                xerr=[entry["sd"] for entry in stats], color=color, marker=marker,
                linestyle="none", capsize=4, label=MODE_LABELS[mode],
            )
        ax.set_title(title)
        ax.set_xlim(0, 1.03)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, labels=STRATEGY_LABELS.values())
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle(f"Final models on the shared holdout\n{spread_label}", fontsize=14)
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)
