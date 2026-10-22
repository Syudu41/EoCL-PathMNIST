"""Run only the four Stage-1 label-flip prototype configurations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import pixel_features, prepare_data, save_splits
from evaluation import bucket_counts, majority_vote, metrics_row
from partitioning import clean_hashes, training_partitions
from poisoning import label_flip
from training import train_members

SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Global seed: {SEED}", flush=True)
    rng = np.random.default_rng(SEED)
    splits = prepare_data(args.data_dir, SEED, rng)
    # Capture clean-image identity before invoking any attack.
    hashes = clean_hashes(splits["train"])
    flip_logs = []
    for name in ("train", "val", "test"):
        splits[name], counts = label_flip(splits[name], rng, name)
        flip_logs.append(counts)
    flips = pd.concat(flip_logs, ignore_index=True)
    flips.to_csv(args.output_dir / "flip_counts.csv", index=False)
    print("Exact label flips by true class:\n" + flips.to_string(index=False), flush=True)
    save_splits(splits, args.output_dir)
    counts = bucket_counts(splits)
    counts.to_csv(args.output_dir / "bucket_counts.csv", index=False)
    for row in counts.itertuples(index=False):
        print(f"Sanity {row.split} bucket={row.bucket}: poisoned={row.poisoned} pristine={row.pristine}", flush=True)
    train_x, test_x = (pixel_features(splits[name]) for name in ("train", "test"))
    train_y = splits["train"]["is_poisoned"].to_numpy()
    test_y = splits["test"]["is_poisoned"].to_numpy()
    test_given = splits["test"]["y_given"].to_numpy()
    rows = []
    with (args.output_dir / "member_parameters.jsonl").open("w") as log:
        for family in ("SVM", "RF"):
            for m in (1, 5):
                partitions = training_partitions(splits["train"], hashes, m)
                members = train_members(train_x, train_y, partitions, family, m, SEED, log)
                predicted = majority_vote(members, test_x, test_given, m)
                rows.append(metrics_row(test_y, predicted, test_given, family, m))
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "results.csv", index=False)
    print("\nDetection metrics (positive = poisoned/reject):", flush=True)
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
