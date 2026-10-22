"""Stable clean-image SHA-256 assignments within given-label buckets."""
import hashlib

import numpy as np


def clean_hashes(frame):
    return [int.from_bytes(hashlib.sha256(image.tobytes(order="C")).digest(), "big")
            for image in frame["image"]]


def training_partitions(frame, hashes, m):
    if len(frame) != len(hashes):
        raise ValueError("Clean hashes must align with training rows")
    buckets = frame["y_given"].to_numpy()
    assignments = np.array([value % m for value in hashes])
    return {(j, i): np.flatnonzero((buckets == j) & (assignments == i))
            for j in range(9) for i in range(m)}
