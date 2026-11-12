import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.dataset import load_dataset, make_train_test_split
from app.provisioning import split_hospitals
from app.study import BASELINE, run_study
from app.study_output import render_plots, write_reports


class StudyTests(unittest.TestCase):
    def test_every_partition_excludes_the_shared_holdout(self):
        _, labels, _ = load_dataset()
        development, test = make_train_test_split(labels, 42)
        for seed in (42, 43, 44):
            for mode in ("iid", "non-iid"):
                with self.subTest(seed=seed, mode=mode):
                    splits = split_hospitals(development, labels, mode, seed)
                    assigned = np.concatenate([indices for pair in splits.values() for indices in pair])
                    np.testing.assert_array_equal(np.sort(assigned), np.sort(development))
                    self.assertEqual(len(np.intersect1d(test, assigned)), 0)

    def test_matrix_uses_shared_final_test_and_exports_aggregate_results(self):
        config = json.loads(Path("configs/study.json").read_text(encoding="utf-8"))
        config.update(seeds=[43], rounds=1, training={"epochs": 1, "batch_size": 256, "learning_rate": 0.01})
        report = run_study(config)
        self.assertEqual(len(report["runs"]), 8)
        self.assertEqual(len(report["partitions"]), 2)
        self.assertEqual(report["test"]["samples"], 114)
        self.assertEqual(len(report["test"]["sha256"]), 64)
        self.assertEqual(report["baseline_training"]["epochs"], 1)
        for run in report["runs"]:
            test = run["test"]
            confusion = test["confusion_matrix"]
            self.assertEqual(test["samples"], 114)
            self.assertEqual(sum(confusion.values()), 114)
            self.assertEqual(confusion["tp"] + confusion["fn"], 42)
            if run["algorithm"] == BASELINE:
                self.assertEqual(run["completed_rounds"], 0)
                self.assertEqual(run["communication"]["total_model_bytes"], 0)
                self.assertNotIn("history", run)
            else:
                self.assertEqual(run["completed_rounds"], 1)
                self.assertNotIn("test", run["history"][0])
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(report, output)
            render_plots(report, output)
            for filename, count in (("runs.csv", 8), ("rounds.csv", 6), ("summary.csv", 8)):
                with (output / filename).open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            for filename in ("validation_curves.png", "test_comparison.png"):
                self.assertTrue((output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_duplicate_study_seeds_are_rejected(self):
        config = json.loads(Path("configs/study.json").read_text(encoding="utf-8"))
        config["seeds"] = [42, 42]
        with self.assertRaisesRegex(ValueError, "distinct"):
            run_study(config)
