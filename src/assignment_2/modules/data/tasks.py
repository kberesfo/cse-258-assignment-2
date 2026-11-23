"""What each task trains on, with its held-out pairs removed.

    t = load_prepped(cfg)
    tables = train_tables(t, "play")     # or "engagement"

A model must never see a pair it is scored on:
  - as a training negative: `exclude` lists every held-out pair, so negative sampling
    skips them;
  - through reviews: "user reviewed X" all but reveals that the user owns and played X,
    so reviews of held-out pairs are dropped here, for every model.
The held-out pairs differ by task, so the filtering is done per task.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Task = Literal["play", "engagement"]
TASKS: tuple[Task, ...] = ("play", "engagement")
KEYS = ["user_id", "item_id"]


@dataclass(frozen=True)
class TrainData:
    """A task's training tables, built by train_tables(t, task)."""

    interactions: pd.DataFrame  # play_train or own_train
    items: pd.DataFrame
    bundles: pd.DataFrame
    reviews: pd.DataFrame  # all reviews minus the task's held-out pairs
    # (user_id, item_id) pairs never to use as training negatives: the task's held-out
    # pairs (play: valid/test candidates, incl. their sampled negatives;
    # engagement: own valid/test).
    exclude: pd.DataFrame | None = None
    n_reviews_dropped: int = 0  # reviews removed because they reveal a held-out pair


def train_tables(t: dict[str, pd.DataFrame], task: Task) -> TrainData:
    """Training tables for a task, with its held-out pairs removed."""
    exclude = held_out_pairs(t, task)
    reviews = drop_pairs(t["reviews"], exclude)
    return TrainData(
        interactions=t["play_train" if task == "play" else "own_train"],
        items=t["items"],
        bundles=t["bundles"],
        reviews=reviews,
        exclude=exclude,
        n_reviews_dropped=len(t["reviews"]) - len(reviews),
    )


def drop_pairs(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows of df whose (user_id, item_id) is not in pairs."""
    merged = df.merge(pairs[KEYS].drop_duplicates(), how="left", on=KEYS, indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def held_out_pairs(t: dict[str, pd.DataFrame], task: Task) -> pd.DataFrame:
    """Every (user, game) a task scores on valid or test."""
    if task == "play":
        # candidates: the held-out games plus their sampled negatives
        parts = [t["play_valid_candidates"], t["play_test_candidates"]]
    else:
        parts = [t["own_valid"], t["own_test"]]
    return pd.concat(parts)[KEYS].drop_duplicates().reset_index(drop=True)
