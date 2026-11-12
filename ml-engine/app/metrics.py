"""Aggregate binary classification metrics with malignant as positive."""

import torch
from torch import nn

from .models import create_model


@torch.no_grad()
def evaluate_weights(
    weights: dict[str, torch.Tensor], features: torch.Tensor, labels: torch.Tensor
) -> dict[str, float | int]:
    model = create_model(0)
    model.load_state_dict(weights)
    model.eval()
    logits = model(features).squeeze(-1)
    predicted = logits >= 0
    positive = labels.bool()
    loss_sum = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
    if not torch.isfinite(loss_sum):
        raise ValueError("Model produced non-finite evaluation loss")
    return {
        "loss_sum": loss_sum.item(),
        "tp": int((predicted & positive).sum().item()),
        "tn": int((~predicted & ~positive).sum().item()),
        "fp": int((predicted & ~positive).sum().item()),
        "fn": int((~predicted & positive).sum().item()),
    }


def summarize_validation(summaries: list[dict[str, float | int]]) -> dict:
    tp, tn, fp, fn = (sum(summary[key] for summary in summaries) for key in ("tp", "tn", "fp", "fn"))
    samples = tp + tn + fp + fn
    return {
        "samples": samples,
        "loss": sum(summary["loss_sum"] for summary in summaries) / samples,
        "accuracy": (tp + tn) / samples,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
