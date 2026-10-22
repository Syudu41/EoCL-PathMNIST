"""Given-label routing, majority rejection, and specified detection metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def majority_vote(members, features, given, m):
    predicted = np.empty(len(given), dtype=np.int64)
    for j in range(9):
        indices = np.flatnonzero(given == j)
        if len(indices):
            votes = np.stack([members[j, i].predict(features[indices]) for i in range(m)])
            predicted[indices] = (votes.sum(axis=0) > m // 2).astype(np.int64)
    return predicted


def metrics_row(truth, predicted, given, family, m):
    row = dict(classifier=family, m=m)
    for scope, mask in [("overall", np.ones(len(truth), dtype=bool))] + [
        (f"class_{j}", given == j) for j in range(9)
    ]:
        if not mask.any():
            raise ValueError(f"Empty test bucket: {scope}")
        actual, decision = truth[mask], predicted[mask]
        row[f"{scope}_accuracy"] = accuracy_score(actual, decision)
        row[f"{scope}_precision"] = precision_score(actual, decision, zero_division=0)
        row[f"{scope}_recall"] = recall_score(actual, decision, zero_division=0)
    return row


def bucket_counts(splits):
    rows = []
    for split, frame in splits.items():
        for j in range(9):
            bucket = frame[frame["y_given"] == j]
            poisoned = int(bucket["is_poisoned"].sum())
            rows.append(dict(split=split, bucket=j, poisoned=poisoned,
                             pristine=len(bucket) - poisoned))
    return pd.DataFrame(rows)
