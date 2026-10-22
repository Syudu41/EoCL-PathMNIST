"""Attack functions operate on the common four-column sample format."""
import numpy as np
import pandas as pd


def label_flip(frame, rng, split):
    poisoned = frame.copy(deep=True)
    counts = []
    for j in range(9):
        indices = np.flatnonzero(frame["y_true"].to_numpy() == j)
        count = len(indices) // 5
        chosen = rng.choice(indices, size=count, replace=False)
        other = rng.integers(0, 8, size=count)
        poisoned.loc[chosen, "y_given"] = other + (other >= j)
        counts.append({"split": split, "y_true": j, "total": len(indices), "flipped": count})
    poisoned["is_poisoned"] = (poisoned["y_given"] != poisoned["y_true"]).astype(np.int64)
    return poisoned, pd.DataFrame(counts)
