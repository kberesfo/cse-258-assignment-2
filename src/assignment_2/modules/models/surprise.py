"""Surprise plumbing shared by the bias and SVD models: negative sampling and bulk scoring."""

from abc import abstractmethod
from typing import Literal, Self, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from surprise import AlgoBase, Dataset, Reader, Trainset

from .base import ScoringModel, TrainData

Target = Literal["owned", "playtime", "minutes"]


def sample_negatives(
    positives: pd.DataFrame,
    exclude: pd.DataFrame | None,
    ratio: int,
    seed: int,
) -> pd.DataFrame:
    """`ratio` negatives per positive, per user: train games the user doesn't own and
    that aren't in `exclude`. Uniform over train games. Returns user_id, item_id."""
    rng = np.random.default_rng(seed)
    users, u_codes = np.unique(positives["user_id"].to_numpy(), return_inverse=True)
    items, i_codes = np.unique(positives["item_id"].to_numpy(), return_inverse=True)
    n_items = len(items)

    # Forbidden pairs as single integers: owned in train, plus excluded pairs.
    forbidden = u_codes.astype(np.int64) * n_items + i_codes
    if exclude is not None and len(exclude):
        eu = pd.Index(users).get_indexer(exclude["user_id"])
        ei = pd.Index(items).get_indexer(exclude["item_id"])
        # pairs outside train can never be sampled anyway
        ok = (eu >= 0) & (ei >= 0)
        forbidden = np.concatenate([forbidden, eu[ok].astype(np.int64) * n_items + ei[ok]])
    forbidden = np.unique(forbidden)

    need = np.bincount(u_codes, minlength=len(users)) * ratio
    room = n_items - np.bincount(forbidden // n_items, minlength=len(users))
    # a user who owns nearly everything gets fewer
    need = np.minimum(need, room)

    chosen = np.empty(0, dtype=np.int64)
    for _ in range(20):  # rejection sampling; each round fills almost all the gap
        have = np.bincount(chosen // n_items, minlength=len(users)) if len(chosen) else 0
        gap = need - have
        if not gap.any():
            break
        draw_u = np.repeat(np.arange(len(users)), gap * 2 + 1)
        draw = draw_u * n_items + rng.integers(0, n_items, len(draw_u))
        draw = draw[~np.isin(draw, forbidden) & ~np.isin(draw, chosen)]
        draw = np.unique(draw)  # also sorts by user
        # Keep at most `gap` new pairs per user.
        du = draw // n_items
        rank = np.arange(len(draw)) - np.searchsorted(du, du)
        chosen = np.concatenate([chosen, draw[rank < gap[du]]])

    return pd.DataFrame({"user_id": users[chosen // n_items], "item_id": items[chosen % n_items]})


class SurpriseModel(ScoringModel):
    """Fits a Surprise algorithm on owned games (plus sampled negatives) and scores rows in bulk.

    Training rows: every owned train game, with target
        "owned":    1                              (task A: will they own it?)
        "playtime": 1 + log1p(playtime_forever)    (task A: owned-but-unplayed stays above 0)
        "minutes":  log1p(playtime_forever)        (task B: what evaluate_engagement scores)
    "owned" / "playtime" add `neg_ratio` sampled non-owned games per positive, target 0.
    "minutes" uses owned games only (neg_ratio = 0): task B only scores games the user owns,
    so a non-owned game at 0 would teach "not owned" = "never played".

    Subclasses implement `_make_algo()` and `_score(u, i)`. Scoring uses the fitted arrays
    directly: much faster than `algo.predict()` per row. Unclipped (no artificial ties),
    except that "minutes" predictions are floored at 0, since log1p(minutes) can't be negative.
    """

    def __init__(
        self,
        target: Target = "playtime",
        neg_ratio: int = 4,
        seed: int = 0,
    ) -> None:
        if target not in get_args(Target):
            raise ValueError(f"target must be one of {get_args(Target)}, got {target!r}")
        if target == "minutes" and neg_ratio != 0:
            raise ValueError('target "minutes" trains on owned games only: set neg_ratio = 0')
        if target != "minutes" and neg_ratio < 1:
            raise ValueError(
                f"target {target!r} needs negatives (neg_ratio >= 1): without them every "
                "training target is > 0"
            )
        self.target, self.neg_ratio, self.seed = target, neg_ratio, seed
        self.algo: AlgoBase | None = None
        self.trainset: Trainset | None = None

    @abstractmethod
    def _make_algo(self) -> AlgoBase: ...

    @abstractmethod
    def _score(self, u: NDArray[np.int64], i: NDArray[np.int64]) -> NDArray[np.float64]: ...

    def _target(self, positives: pd.DataFrame) -> NDArray[np.float64]:
        minutes = np.log1p(positives["playtime_forever"].to_numpy(np.float64))
        match self.target:
            case "owned":
                return np.ones(len(positives))
            case "playtime":
                return 1 + minutes
            case "minutes":
                return minutes

    def _training_rows(self, data: TrainData) -> pd.DataFrame:
        pos = data.interactions
        rows = pos[["user_id", "item_id"]].assign(target=self._target(pos))
        if self.neg_ratio == 0:
            return rows.reset_index(drop=True)
        neg = sample_negatives(pos, data.exclude, self.neg_ratio, self.seed)
        return pd.concat([rows, neg.assign(target=0.0)], ignore_index=True)

    def fit(self, data: TrainData) -> Self:
        rows = self._training_rows(data)
        reader = Reader(rating_scale=(float(rows.target.min()), float(rows.target.max())))
        self.trainset = Dataset.load_from_df(rows, reader).build_full_trainset()
        self.algo = self._make_algo()
        self.algo.fit(self.trainset)
        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.algo is None or self.trainset is None:
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        u = rows["user_id"].map(self.trainset._raw2inner_id_users)
        i = rows["item_id"].map(self.trainset._raw2inner_id_items)
        # Candidates only contain train users and items, so a miss is a pipeline bug.
        missing = u.isna() | i.isna()
        if missing.any():
            raise ValueError(
                f"{int(missing.sum()):,} rows have users/items not seen in training, "
                f"e.g. {rows.loc[missing, ['user_id', 'item_id']].iloc[0].tolist()}"
            )
        pred = self._score(u.to_numpy(np.int64), i.to_numpy(np.int64))
        # log1p(minutes) can't be negative; clipping only removes impossible predictions
        return np.maximum(pred, 0.0) if self.target == "minutes" else pred
