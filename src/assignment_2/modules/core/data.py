"""Turn a raw dataset into cleaned, split tables that every model shares."""

import json
import shutil

import numpy as np
import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core.datasets import LOADERS
from assignment_2.modules.core.features import ranking_candidates

SPLITS = ("train", "valid", "test")
TARGETS = {"rating": "rating", "binary": "label", "ranking": None}
MIN_INTERACTIONS_TO_HOLD_OUT = 3


def target_column(cfg: Config) -> str | None:
    """The column models predict for this task (None for ranking)."""
    return TARGETS[cfg.task]


def standardize(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Check a loader's output and drop rows that are unusable for this task."""
    target = TARGETS[task]
    required = ["user", "item"] + ([target] if target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"task {task!r} needs columns {missing}, which the loader didn't return")

    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"[prep] dropped {before - len(df):,} rows with missing {required}")
    if target == "label" and not df["label"].isin([0, 1]).all():
        raise ValueError("the label column must contain only 0 and 1")
    return df.astype({"user": str, "item": str})


def random_split(
    df: pd.DataFrame, valid_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle rows, then cut them into train, valid, and test."""
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test, n_valid = int(len(df) * test_frac), int(len(df) * valid_frac)
    test_idx, valid_idx = idx[:n_test], idx[n_test : n_test + n_valid]
    train_idx = idx[n_test + n_valid :]
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]


def _held_out_count(sizes: np.ndarray, frac: float) -> np.ndarray:
    """At least one interaction per eligible user, so sparse users still get evaluated."""
    if frac == 0:
        return np.zeros_like(sizes)
    counts = np.maximum(np.round(sizes * frac).astype(int), 1)
    return np.where(sizes >= MIN_INTERACTIONS_TO_HOLD_OUT, counts, 0)


def temporal_split(
    df: pd.DataFrame, valid_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out each user's latest interactions: newest for test, the next newest for valid.

    Users with fewer than MIN_INTERACTIONS_TO_HOLD_OUT interactions stay in train.
    """
    ordered = df.sort_values("timestamp", kind="stable")
    users = ordered.groupby("user", sort=False)["user"]
    sizes = users.transform("size").to_numpy()
    from_end = users.cumcount(ascending=False).to_numpy()

    n_test = _held_out_count(sizes, test_frac)
    n_valid = np.minimum(_held_out_count(sizes, valid_frac), np.maximum(sizes - n_test - 1, 0))

    is_test = from_end < n_test
    is_valid = ~is_test & (from_end < n_test + n_valid)
    return ordered[~is_test & ~is_valid], ordered[is_valid], ordered[is_test]


def _drop_cold(train: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    known = held_out["user"].isin(train["user"]) & held_out["item"].isin(train["item"])
    return held_out[known]


def prep(cfg: Config) -> None:
    """Load the dataset named in the config, split it, and save the splits."""
    if cfg.dataset not in LOADERS:
        options = ", ".join(LOADERS)
        raise ValueError(f"dataset {cfg.dataset!r} has no loader in LOADERS (options: {options})")

    df = standardize(LOADERS[cfg.dataset](cfg), cfg.task)
    s = cfg.split
    if s.by_time and "timestamp" in df.columns:
        strategy = "temporal"
        train, valid, test = temporal_split(df, s.valid_frac, s.test_frac)
    else:
        strategy = "random"
        if s.by_time:
            print("[prep] no timestamp column, so using a random split")
        train, valid, test = random_split(df, s.valid_frac, s.test_frac, cfg.seed)

    if s.drop_cold:
        counts = len(valid), len(test)
        valid, test = _drop_cold(train, valid), _drop_cold(train, test)
        print(
            f"[prep] dropped cold rows: valid {counts[0] - len(valid):,}, test {counts[1] - len(test):,}"
        )
    if len(valid) == 0 or len(test) == 0:
        raise ValueError("valid or test is empty; check the split settings")

    # Predictions from an earlier split no longer line up, so never keep them.
    shutil.rmtree(cfg.paths.preds, ignore_errors=True)
    cfg.paths.preds.mkdir(parents=True)

    for name, frame in zip(SPLITS, (train, valid, test)):
        frame.reset_index(drop=True).to_pickle(cfg.paths.splits / f"{name}.pkl")
    if cfg.task == "ranking":
        everything = pd.concat([train, valid, test])
        e = cfg.evaluation
        for offset, (name, frame) in enumerate((("valid", valid), ("test", test))):
            table = ranking_candidates(frame, everything, e.n_negatives, cfg.seed + offset)
            table.to_pickle(cfg.paths.splits / f"{name}_candidates.pkl")

    meta = {
        "dataset": cfg.dataset,
        "task": cfg.task,
        "strategy": strategy,
        "users": int(df["user"].nunique()),
        "items": int(df["item"].nunique()),
        "interactions": len(df),
        "sparsity": 1 - len(df) / (df["user"].nunique() * df["item"].nunique()),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }
    (cfg.paths.splits / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[prep] {cfg.dataset}: {strategy} split, {len(train):,} / {len(valid):,} / {len(test):,}"
    )


def _read(cfg: Config, filename: str) -> pd.DataFrame:
    path = cfg.paths.splits / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make prep`")
    return pd.read_pickle(path)


def load_split(cfg: Config, name: str) -> pd.DataFrame:
    """Load train, valid, or test."""
    return _read(cfg, f"{name}.pkl")


def eval_table(cfg: Config, split: str) -> pd.DataFrame:
    """The rows every model must score for a split, in a fixed order.

    For rating and binary tasks this is the split. For ranking it is the candidate table.
    """
    return _read(cfg, f"{split}_candidates.pkl" if cfg.task == "ranking" else f"{split}.pkl")


def summary(cfg: Config) -> dict:
    """Dataset and split statistics recorded by prep."""
    path = cfg.paths.splits / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make prep`")
    return json.loads(path.read_text())
