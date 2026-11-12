import json
import math
import subprocess
import sys
import unittest
from dataclasses import fields

import numpy as np
import torch

from app.clients import HospitalClient, LocalTrainingConfig
from app.dataset import load_dataset, make_train_test_split
from app.models import create_model
from app.partitioning import HOSPITAL_IDS, partition_hospitals, split_hospital


class DataTests(unittest.TestCase):
    def setUp(self):
        self.features, self.labels, self.names = load_dataset()
        self.train, self.test = make_train_test_split(self.labels, 42)

    def test_dataset_and_stratified_split(self):
        self.assertEqual(self.features.shape, (569, 30))
        self.assertTrue(np.isfinite(self.features).all())
        self.assertEqual(len(set(self.names)), 30)
        np.testing.assert_array_equal(np.bincount(self.labels), [357, 212])
        self.assertEqual((len(self.train), len(self.test)), (455, 114))
        np.testing.assert_array_equal(np.bincount(self.labels[self.test]), [72, 42])
        np.testing.assert_array_equal(
            np.sort(np.concatenate((self.train, self.test))), np.arange(569)
        )
        for actual, expected in zip(make_train_test_split(self.labels, 42), (self.train, self.test)):
            np.testing.assert_array_equal(actual, expected)

    def test_partitions_are_disjoint_complete_and_repeatable(self):
        for mode in ("iid", "non-iid"):
            with self.subTest(mode=mode):
                partitions = partition_hospitals(self.train, self.labels, mode, 42)
                self.assertEqual(tuple(partitions), HOSPITAL_IDS)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate(list(partitions.values()))), np.sort(self.train)
                )
                repeat = partition_hospitals(self.train, self.labels, mode, 42)
                for hospital, indices in partitions.items():
                    np.testing.assert_array_equal(indices, repeat[hospital])
                    self.assertFalse(np.intersect1d(indices, self.test).size)
                    train, validation = split_hospital(indices, self.labels, 42)
                    np.testing.assert_array_equal(
                        np.sort(np.concatenate((train, validation))), np.sort(indices)
                    )
                    self.assertEqual(set(self.labels[train]), {0, 1})
                    self.assertEqual(set(self.labels[validation]), {0, 1})

    def test_partition_modes_produce_distinct_class_distributions(self):
        iid = partition_hospitals(self.train, self.labels, "iid", 42)
        skewed = partition_hospitals(self.train, self.labels, "non-iid", 42)
        iid_rates = [self.labels[indices].mean() for indices in iid.values()]
        skewed_rates = [self.labels[indices].mean() for indices in skewed.values()]
        self.assertLess(max(iid_rates) - min(iid_rates), 0.02)
        self.assertGreater(skewed_rates[0] - skewed_rates[2], 0.40)


class TrainingTests(unittest.TestCase):
    def test_validation_is_transformed_with_training_statistics(self):
        train = np.tile([[0.0], [2.0]], (1, 30))
        validation = np.full((2, 30), 101.0)
        client = HospitalClient("A", train, np.array([0, 1]), validation, np.array([0, 1]))
        torch.testing.assert_close(client._train.tensors[0][:, 0], torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(client._validation[0], torch.full((2, 30), 100.0))

    def test_one_sgd_step_matches_binary_logistic_gradient(self):
        train = np.tile([[-1.0], [1.0]], (1, 30))
        labels = np.array([0, 1])
        client = HospitalClient("A", train, labels, train.copy(), labels.copy())
        initial = {name: torch.zeros_like(value) for name, value in create_model(42).state_dict().items()}
        config = LocalTrainingConfig(epochs=1, batch_size=2, learning_rate=0.1)
        update = client.train_local(initial, config, round_index=0, seed=42)
        # At zero logits, the full-batch BCE gradient is -0.5 per weight and 0 for bias.
        torch.testing.assert_close(update.model_state["weight"], torch.full((1, 30), 0.05))
        torch.testing.assert_close(update.model_state["bias"], torch.zeros(1))
        self.assertFalse(initial["weight"].count_nonzero().item())
        self.assertEqual(update.num_train_samples, 2)
        self.assertEqual(
            {field.name for field in fields(update)},
            {"hospital_id", "round_index", "num_train_samples", "model_state", "summary_metrics"},
        )

    def test_seeded_minibatch_training_is_repeatable(self):
        features, labels, _ = load_dataset()
        development, _ = make_train_test_split(labels, 42)
        site = partition_hospitals(development, labels, "iid", 42)["A"]
        train, validation = split_hospital(site, labels, 42)
        client = HospitalClient("A", features[train], labels[train], features[validation], labels[validation])
        initial = create_model(42).state_dict()
        config = LocalTrainingConfig(epochs=2, batch_size=16, learning_rate=0.01)
        first = client.train_local(initial, config, round_index=0, seed=42)
        second = client.train_local(initial, config, round_index=0, seed=42)
        self.assertEqual(first.summary_metrics, second.summary_metrics)
        for name in initial:
            torch.testing.assert_close(first.model_state[name], second.model_state[name], rtol=0, atol=0)

    def test_invalid_local_training_settings(self):
        for args in ((0, 16, 0.01), (1, 0, 0.01), (1, 16, -0.01), (1, 16, math.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                LocalTrainingConfig(*args)


class CliTests(unittest.TestCase):
    def test_cli_reports_only_counts_and_aggregate_metrics(self):
        result = subprocess.run(
            [sys.executable, "-m", "app", "--config", "configs/iid.json"],
            capture_output=True, text=True, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(
            set(report),
            {"mode", "seed", "model", "training", "development_samples", "reserved_test_samples", "hospitals"},
        )
        self.assertEqual(report["reserved_test_samples"], 114)
        self.assertEqual(tuple(report["hospitals"]), HOSPITAL_IDS)
        total = 0
        for site in report["hospitals"].values():
            self.assertEqual(set(site), {"train", "validation"})
            self.assertEqual(set(site["train"]), {"samples", "benign", "malignant"})
            self.assertEqual(
                set(site["validation"]), {"samples", "benign", "malignant", "loss", "accuracy"}
            )
            for split in site.values():
                self.assertEqual(split["samples"], split["benign"] + split["malignant"])
                total += split["samples"]
            self.assertTrue(math.isfinite(site["validation"]["loss"]))
        self.assertEqual(total, 455)


if __name__ == "__main__":
    unittest.main()
