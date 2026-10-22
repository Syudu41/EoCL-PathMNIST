"""PathMNIST pool and stratified splits; labels remain metadata."""
from pathlib import Path

import numpy as np
import pandas as pd
from medmnist import PathMNIST
from sklearn.model_selection import train_test_split

COLUMNS = ["image", "y_true", "y_given", "is_poisoned"]


def prepare_data(root, seed, rng):
    Path(root).mkdir(parents=True, exist_ok=True)
    dataset = PathMNIST(split="train", root=str(root), size=28, download=True)
    indices = rng.choice(len(dataset.imgs), size=40_000, replace=False)
    labels = dataset.labels.reshape(-1)
    train, held = train_test_split(
        indices, train_size=30_000, stratify=labels[indices], random_state=seed
    )
    val, test = train_test_split(
        held, train_size=5_000, stratify=labels[held], random_state=seed
    )
    splits = {}
    for name, selected in (("train", train), ("val", val), ("test", test)):
        images = dataset.imgs[selected]
        if images.shape[1:] != (28, 28, 3) or images.dtype != np.uint8:
            raise ValueError("Expected clean uint8 PathMNIST images of shape (28,28,3)")
        truth = labels[selected].astype(np.int64)
        splits[name] = pd.DataFrame({
            "image": list(images), "y_true": truth,
            "y_given": truth.copy(), "is_poisoned": np.zeros(len(truth), dtype=np.int64),
        }, columns=COLUMNS)
    return splits


def pixel_features(frame):
    """Explicit feature allowlist: image pixels only, scaled to [0,1]."""
    return np.stack(frame["image"].to_numpy()).reshape(len(frame), 2352).astype(np.float32) / 255.0


def save_splits(splits, output):
    """Pickle preserves image arrays and exactly the four specified columns."""
    for name, frame in splits.items():
        frame[COLUMNS].to_pickle(Path(output) / f"{name}.pkl")
