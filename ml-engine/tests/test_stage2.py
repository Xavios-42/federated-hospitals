import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from app.__main__ import run_federated
from app.algorithms import Strategy
from app.clients import ClientUpdate, HospitalClient, LocalTrainingConfig
from app.coordinator import summarize_validation
from app.models import create_model


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.state = {"weight": torch.tensor([10.0, -1.0])}
        self.updates = [
            ClientUpdate("A", 1, 1, {"weight": torch.tensor([8.0, 5.0])}, {}),
            ClientUpdate("B", 1, 2, {"weight": torch.tensor([12.0, 2.0])}, {}),
            ClientUpdate("C", 1, 5, {"weight": torch.tensor([14.0, -3.0])}, {}),
        ]

    def test_weighted_average_for_fedavg_and_fedprox(self):
        for strategy in (Strategy("fedavg"), Strategy("fedprox", mu=0.1)):
            with self.subTest(algorithm=strategy.algorithm):
                state, velocity = strategy.aggregate(self.state, self.updates[::-1], 1)
                torch.testing.assert_close(state["weight"], torch.tensor([12.75, -0.75]))
                self.assertIsNone(velocity)

    def test_fedavgm_with_zero_momentum_matches_fedavg(self):
        avg, _ = Strategy("fedavg").aggregate(self.state, self.updates, 1)
        momentum, _ = Strategy("fedavgm").aggregate(self.state, self.updates, 1)
        torch.testing.assert_close(momentum["weight"], avg["weight"])

    def test_server_momentum_persists_and_resets_between_runs(self):
        strategy = Strategy("fedavgm", beta=0.5, server_lr=0.5)
        first, velocity = strategy.aggregate(self.state, self.updates, 1)
        torch.testing.assert_close(velocity["weight"], torch.tensor([2.75, 0.25]))
        torch.testing.assert_close(first["weight"], torch.tensor([11.375, -0.875]))
        second_updates = [
            replace(update, round_index=2, model_state={"weight": first["weight"] + 2})
            for update in self.updates
        ]
        second, second_velocity = strategy.aggregate(first, second_updates, 2, velocity)
        torch.testing.assert_close(second_velocity["weight"], torch.tensor([3.375, 2.125]))
        torch.testing.assert_close(second["weight"], torch.tensor([13.0625, 0.1875]))
        restarted, _ = strategy.aggregate(first, second_updates, 2)
        torch.testing.assert_close(restarted["weight"], first["weight"] + 1)
        torch.testing.assert_close(velocity["weight"], torch.tensor([2.75, 0.25]))
        torch.testing.assert_close(self.state["weight"], torch.tensor([10.0, -1.0]))

    def test_malformed_or_incomplete_rounds_are_rejected(self):
        a, b, c = self.updates
        malformed = [
            [a, b],
            [a, b, a],
            [a, b, replace(c, hospital_id="D")],
            [a, b, replace(c, round_index=0)],
            [a, b, replace(c, num_train_samples=0)],
            [a, b, replace(c, model_state={})],
            [a, b, replace(c, model_state={"weight": torch.zeros(1)})],
            [a, b, replace(c, model_state={"weight": torch.zeros(2, dtype=torch.float64)})],
            [a, b, replace(c, model_state={"weight": torch.tensor([float("nan"), 0.0])})],
            [a, b, replace(c, model_state={"weight": torch.tensor([float("inf"), 0.0])})],
        ]
        for index, updates in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                Strategy("fedavg").aggregate(self.state, updates, 1)
        torch.testing.assert_close(self.state["weight"], torch.tensor([10.0, -1.0]))

    def test_invalid_strategy_settings(self):
        for settings in (
            {"algorithm": "unknown"},
            {"algorithm": "fedavg", "mu": 0.1},
            {"algorithm": "fedprox", "mu": -0.1},
            {"algorithm": "fedprox", "beta": 0.5},
            {"algorithm": "fedavgm", "beta": 1.0},
            {"algorithm": "fedavgm", "server_lr": math.inf},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Strategy(**settings)


class ClientAndRunTests(unittest.TestCase):
    def test_fedprox_zero_matches_plain_sgd_and_positive_mu_changes_gradient(self):
        features = np.zeros((2, 30))
        features[:, 0] = [-1, 1]
        labels = np.array([0, 1])
        client = HospitalClient("A", features, labels, features + 0.5, labels)
        state = {name: torch.zeros_like(value) for name, value in create_model(42).state_dict().items()}
        config = LocalTrainingConfig(epochs=2, batch_size=2, learning_rate=0.1)
        plain = client.train_local(state, config, round_index=1, seed=42)
        zero = client.train_local(state, config, round_index=1, seed=42, mu=0)
        prox = client.train_local(state, config, round_index=1, seed=42, mu=0.5)
        for name in state:
            torch.testing.assert_close(zero.model_state[name], plain.model_state[name], rtol=0, atol=0)
        # The first step sets the weight to 0.05. The second adds mu * 0.05 to its gradient.
        expected = 0.05 - 0.1 * (1 / (1 + math.exp(-0.05)) - 1 + 0.5 * 0.05)
        self.assertAlmostEqual(prox.model_state["weight"][0, 0].item(), expected, places=7)
        self.assertLess(prox.model_state["weight"][0, 0], plain.model_state["weight"][0, 0])

    def test_validation_metrics_pool_counts_and_sample_weight_loss(self):
        metrics = summarize_validation([
            {"tp": 1, "tn": 1, "fp": 0, "fn": 0, "loss_sum": 2.0},
            {"tp": 1, "tn": 4, "fp": 1, "fn": 2, "loss_sum": 4.0},
        ])
        self.assertEqual(metrics["samples"], 10)
        self.assertAlmostEqual(metrics["loss"], 0.6)
        self.assertAlmostEqual(metrics["accuracy"], 0.7)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        self.assertEqual(metrics["confusion_matrix"], {"tp": 2, "tn": 5, "fp": 1, "fn": 2})

    def test_round_history_model_transfer_totals_and_repeatability(self):
        training = LocalTrainingConfig(epochs=1, batch_size=64, learning_rate=0.01)
        settings = {"algorithm": "fedavgm", "rounds": 2, "beta": 0.9}
        report = run_federated("iid", 42, training, settings)
        self.assertEqual(report, run_federated("iid", 42, training, settings))
        self.assertEqual(report["reserved_test_samples"], 114)
        self.assertEqual(report["hospital_training_samples"], {"A": 120, "B": 121, "C": 121})
        self.assertEqual(report["completed_rounds"], 2)
        self.assertEqual([entry["round"] for entry in report["history"]], [1, 2])
        comm = report["communication"]
        for entry in report["history"]:
            self.assertEqual(entry["validation"]["samples"], 93)
            self.assertGreater(entry["upload_model_bytes"], 0)
            self.assertGreater(entry["download_model_bytes"], 0)
        self.assertGreater(comm["initial_download_model_bytes"], 0)
        self.assertEqual(comm["upload_model_bytes"], sum(e["upload_model_bytes"] for e in report["history"]))
        self.assertEqual(
            comm["download_model_bytes"],
            comm["initial_download_model_bytes"] + sum(e["download_model_bytes"] for e in report["history"]),
        )
        self.assertEqual(comm["total_model_bytes"], comm["upload_model_bytes"] + comm["download_model_bytes"])
