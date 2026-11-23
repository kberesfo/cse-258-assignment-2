"""Similarity (kNN) baselines for task A: item-based and user-based collaborative filtering.

Both build a 0/1 user × game matrix of training positives, compute similarities with
`core/similarity.py`, and score each candidate from the user's own history:

    item kNN: max similarity between the candidate and any game the user played
    user kNN: sum of the user's similarity to every user who played the candidate

Unknown users or candidates (no positives in training) score 0.
"""

from typing import Literal, Self, get_args

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from assignment_2.modules.core import (
    SimMetric,
    interaction_matrix,
    item_similarity,
    user_similarity,
)
from assignment_2.modules.data.tasks import TrainData

from ..base import ScoringModel

Positives = Literal["played", "owned"]


def _positives(data: TrainData, positives: Positives, min_minutes: int) -> pd.DataFrame:
    """Training rows that count as positives: games played ≥ min_minutes, or every owned game."""
    rows = data.interactions
    if positives == "played":
        rows = rows[rows.playtime_forever >= min_minutes]
    return rows


class ItemKNNRanker(ScoringModel):
    """Item kNN: a candidate's max similarity to the games the user played."""

    def __init__(
        self,
        metric: SimMetric = "jaccard",
        positives: Positives = "played",
        min_minutes: int = 120,
    ) -> None:
        if metric not in get_args(SimMetric):
            raise ValueError(f"metric must be one of {get_args(SimMetric)}, got {metric!r}")
        if positives not in ("played", "owned"):
            raise ValueError(f"positives must be 'played' or 'owned', got {positives!r}")
        self.metric, self.positives, self.min_minutes = metric, positives, min_minutes
        # set by fit()
        self.X: sp.csr_matrix | None = None  # user × game positives
        self.S: sp.csr_matrix | None = None  # game × game similarity
        self.users: pd.Index | None = None
        self.items: pd.Index | None = None

    def fit(self, data: TrainData) -> Self:
        self.X, self.users, self.items = interaction_matrix(
            _positives(data, self.positives, self.min_minutes)
        )
        self.S = item_similarity(self.X, self.metric)
        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.S is None:
            raise RuntimeError("ItemKNN.predict called before fit")
        u_idx = self.users.get_indexer(rows.user_id)  # -1 = user with no positives
        i_idx = self.items.get_indexer(rows.item_id)  # -1 = candidate nobody played
        scores = np.zeros(len(rows))

        # One sparse slice per user (candidates × that user's history), not one per row.
        groups = pd.Series(np.arange(len(rows))).groupby(u_idx).groups
        for u, positions in groups.items():
            if u < 0:
                continue
            positions = np.asarray(positions)
            ok = positions[i_idx[positions] >= 0]
            history = self.X[u].indices
            if len(ok) == 0 or len(history) == 0:
                continue
            sims = self.S[i_idx[ok]][:, history]
            scores[ok] = sims.max(axis=1).toarray().ravel()
        return scores


class UserKNNRanker(ScoringModel):
    """User kNN: the sum of the user's similarity to every user who played the candidate."""

    def __init__(
        self,
        metric: SimMetric = "cosine",
        positives: Positives = "played",
        min_minutes: int = 120,
    ) -> None:
        if metric not in get_args(SimMetric):
            raise ValueError(f"metric must be one of {get_args(SimMetric)}, got {metric!r}")
        if positives not in ("played", "owned"):
            raise ValueError(f"positives must be 'played' or 'owned', got {positives!r}")
        self.metric, self.positives, self.min_minutes = metric, positives, min_minutes

        # set by fit()
        self.R: sp.csr_matrix | None = None  # R[u, i] = Σ over users v who played i of sim(u, v)
        self.users: pd.Index | None = None
        self.items: pd.Index | None = None

    def fit(self, data: TrainData) -> Self:
        X, self.users, self.items = interaction_matrix(
            _positives(data, self.positives, self.min_minutes)
        )
        self.R = (user_similarity(X, self.metric) @ X).tocsr()
        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.R is None:
            raise RuntimeError("UserKNN.predict called before fit")
        u_idx = self.users.get_indexer(rows.user_id)
        i_idx = self.items.get_indexer(rows.item_id)
        ok = (u_idx >= 0) & (i_idx >= 0)
        scores = np.zeros(len(rows))
        scores[ok] = np.asarray(self.R[u_idx[ok], i_idx[ok]]).ravel()
        return scores
