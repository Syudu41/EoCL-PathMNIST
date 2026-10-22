"""Checks for attack accounting, feature leakage, hashing, and voting."""
import unittest
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from data_prep import pixel_features
from evaluation import majority_vote, metrics_row
from partitioning import clean_hashes, training_partitions
from poisoning import label_flip
from training import train_members


class ConstantMember:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        labels = np.repeat(np.arange(9), 11)
        self.frame = pd.DataFrame({
            "image": [np.full((28, 28, 3), i, dtype=np.uint8) for i in range(99)],
            "y_true": labels, "y_given": labels.copy(), "is_poisoned": 0,
        })

    def test_flips_floor_and_reproducibility(self):
        first, counts = label_flip(self.frame, np.random.default_rng(42), "train")
        second, _ = label_flip(self.frame, np.random.default_rng(42), "train")
        np.testing.assert_array_equal(first.y_given, second.y_given)
        self.assertTrue((counts.flipped == 2).all())
        self.assertEqual(int(first.is_poisoned.sum()), 18)
        np.testing.assert_array_equal(first.is_poisoned, first.y_given != first.y_true)
        np.testing.assert_array_equal(np.stack(first.image), np.stack(self.frame.image))
        self.assertEqual(int(self.frame.is_poisoned.sum()), 0)

    def test_hashes_partitions_and_feature_allowlist(self):
        import hashlib
        poisoned, _ = label_flip(self.frame, np.random.default_rng(42), "train")
        hashes = clean_hashes(self.frame)
        self.assertEqual(hashes, clean_hashes(poisoned))
        self.assertEqual(hashes[0], int(hashlib.sha256(self.frame.image[0].tobytes()).hexdigest(), 16))
        partitions = training_partitions(poisoned, hashes, 5)
        all_indices = np.concatenate(list(partitions.values()))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(99))
        for (j, i), indices in partitions.items():
            self.assertTrue((poisoned.y_given.iloc[indices] == j).all())
            self.assertTrue(all(hashes[k] % 5 == i for k in indices))
        np.testing.assert_array_equal(pixel_features(self.frame), pixel_features(poisoned))
        self.assertEqual(pixel_features(poisoned).shape, (99, 2352))
        self.assertAlmostEqual(float(pixel_features(poisoned).max()), 98 / 255)

    def test_all_members_vote_and_metrics_use_given_bucket(self):
        given = np.repeat(np.arange(9), 2)
        members = {(j, i): ConstantMember(int(i < 3)) for j in range(9) for i in range(5)}
        predicted = majority_vote(members, np.zeros((18, 2352)), given, 5)
        np.testing.assert_array_equal(predicted, np.ones(18))
        truth = np.tile([0, 1], 9)
        row = metrics_row(truth, predicted, given, "SVM", 5)
        self.assertEqual(row["overall_accuracy"], 0.5)
        self.assertEqual(row["class_4_precision"], 0.5)
        self.assertEqual(row["class_4_recall"], 1)
        zeros = metrics_row(truth, np.zeros(18), given, "SVM", 1)
        self.assertEqual(zeros["class_4_precision"], 0)
        self.assertEqual(zeros["class_4_recall"], 0)

    def test_svm_scaler_fits_only_own_partition(self):
        features = np.arange(72, dtype=np.float32).reshape(36, 2)
        target = np.tile([0, 1, 0, 1], 9)
        partitions = {(j, 0): np.arange(j * 4, j * 4 + 4) for j in range(9)}
        with redirect_stdout(io.StringIO()):
            members = train_members(features, target, partitions, "SVM", 1, 42, io.StringIO())
        for (j, i), model in members.items():
            np.testing.assert_allclose(model[0].mean_, features[partitions[j, i]].mean(axis=0))
            self.assertEqual(model[0].n_samples_seen_, 4)
            self.assertEqual(model[1].C, 1)
            self.assertEqual(model[1].class_weight, "balanced")


if __name__ == "__main__":
    unittest.main()
