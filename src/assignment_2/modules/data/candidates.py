"""Fixed ranking candidates: sampled once in prep so every model ranks the same games.

One row per (user, candidate game). For each user with held-out games in a split:
    label = 1   each of the user's held-out games in that split
    label = 0   n_negatives games sampled from train items the user owns in *no* split

Every model scores exactly these rows, so differences between models come from the
models, not from which negatives each happened to draw. Metrics with several positives
per user (Recall@k, NDCG@k) are computed per user over this table.

    negatives = "uniform"     each train item equally likely (SASRec-style protocol)
    negatives = "popularity"  weighted by train popularity (harder; popular decoys)
"""

from typing import Literal

import numpy as np
import pandas as pd


def ranking_candidates(
    held_out: pd.DataFrame,
    train: pd.DataFrame,
    owned: pd.DataFrame,
    n_negatives: int,
    seed: int,
    negatives: Literal["uniform", "popularity"] = "uniform",
) -> pd.DataFrame:
    """Candidate table for one split: user_id, item_id, label (int8)."""
    rng = np.random.default_rng(seed)

    pop = train["item_id"].value_counts()
    pool = pop.index.to_numpy()
    p = (pop / pop.sum()).to_numpy() if negatives == "popularity" else None
    index = {item: i for i, item in enumerate(pool)}

    # Every game each user owns in any split (never a negative).
    owned_by = owned.groupby("user_id")["item_id"].agg(set).to_dict()
    positives = held_out.groupby("user_id")["item_id"].agg(list)

    users, items, labels = [], [], []
    for user, pos in positives.items():
        excluded = np.fromiter(
            (index[i] for i in owned_by.get(user, ()) if i in index), dtype=np.int64
        )
        n_avail = len(pool) - len(np.unique(excluded))
        n = min(n_negatives, n_avail)
        neg = _sample_excluding(rng, len(pool), n, excluded, p)

        users.extend([user] * (len(pos) + n))
        items.extend(pos)
        items.extend(pool[neg])
        labels.extend([1] * len(pos) + [0] * n)

    return pd.DataFrame(
        {"user_id": users, "item_id": items, "label": np.asarray(labels, dtype=np.int8)}
    )


def _sample_excluding(
    rng: np.random.Generator,
    n_pool: int,
    n: int,
    excluded: np.ndarray,
    p: np.ndarray | None,
) -> np.ndarray:
    """n distinct indices in [0, n_pool) not in `excluded`, drawn with weights p."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    bad = np.zeros(n_pool, dtype=bool)
    bad[excluded] = True
    chosen: list[int] = []
    taken = bad.copy()
    # Rejection sampling: cheap because a user owns a small share of all games.
    for _ in range(50):
        draw = rng.choice(n_pool, size=2 * (n - len(chosen)) + 8, p=p)
        for i in draw:
            if not taken[i]:
                taken[i] = True
                chosen.append(int(i))
                if len(chosen) == n:
                    return np.asarray(chosen, dtype=np.int64)
    # Rare fallback (heavy users with popularity weights): fill uniformly from what's left.
    rest = np.flatnonzero(~taken)
    fill = rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False)
    return np.asarray(chosen + fill.tolist(), dtype=np.int64)
