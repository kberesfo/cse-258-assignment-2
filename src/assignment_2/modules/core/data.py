"""Data loading, validation splits, and prediction-file writing (shared by all tasks)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID_DTYPES = {"user": str, "item": str, "item1": str, "item2": str}

PathLike = str | Path


def data_path(name: str, data_dir: PathLike | None = None) -> Path:
    """Resolve a data file: explicit data_dir > $A1_DATA_DIR > ./data"""
    return Path(data_dir or os.environ.get("A1_DATA_DIR", "data")) / name


def load_csv(name: str, data_dir: PathLike | None = None) -> pd.DataFrame:
    """Load a train/test CSV with user/item IDs kept as strings (never parsed as numbers)."""
    path = data_path(name, data_dir)
    header = pd.read_csv(path, nrows=0).columns

    return pd.read_csv(path, dtype={c: t for c, t in ID_DTYPES.items() if c in header})


def random_split(
    df: pd.DataFrame, val_frac: float = 0.25, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle rows and hold out val_frac as a validation set. Fixed seed = same split every run.

    Returns (train, val).
    """
    idx = np.random.default_rng(seed).permutation(len(df))
    n_val = int(len(df) * val_frac)
    train = df.iloc[idx[n_val:]].reset_index(drop=True)
    val = df.iloc[idx[:n_val]].reset_index(drop=True)
    return train, val
