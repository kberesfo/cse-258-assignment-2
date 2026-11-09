"""Feature construction shared across models. Everything here works on the standard table."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class IdIndex:
    """Maps user and item IDs to consecutive integers. Unknown IDs map to -1."""

    users: pd.Index
    items: pd.Index

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "IdIndex":
        return cls(pd.Index(df["user"].unique()), pd.Index(df["item"].unique()))

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def user_codes(self, users: pd.Series) -> np.ndarray:
        return self.users.get_indexer(users)

    def item_codes(self, items: pd.Series) -> np.ndarray:
        return self.items.get_indexer(items)


def interaction_matrix(
    df: pd.DataFrame, index: IdIndex, values: str | None = None
) -> sparse.csr_matrix:
    """Users-by-items CSR matrix, which is what `implicit` trains on.

    Without `values` the entries are 1. Rows with IDs outside `index` are ignored.
    """
    users, items = index.user_codes(df["user"]), index.item_codes(df["item"])
    known = (users >= 0) & (items >= 0)
    data = np.ones(known.sum()) if values is None else df[values].to_numpy(float)[known]
    shape = (index.n_users, index.n_items)
    matrix = sparse.csr_matrix((data, (users[known], items[known])), shape=shape)
    if values is None:
        matrix.data = np.ones_like(matrix.data)  # repeat interactions were summed
    return matrix


def user_items(df: pd.DataFrame) -> dict[str, set[str]]:
    """The set of items each user interacted with."""
    return df.groupby("user")["item"].agg(set).to_dict()


def item_users(df: pd.DataFrame) -> dict[str, set[str]]:
    """The set of users who interacted with each item."""
    return df.groupby("item")["user"].agg(set).to_dict()


def surprise_trainset(df: pd.DataFrame, target: str = "rating"):
    """Build the full trainset object that `surprise` models fit on."""
    from surprise import Dataset, Reader

    scale = (df[target].min(), df[target].max())
    frame = df[["user", "item", target]].astype({"user": object, "item": object})
    return Dataset.load_from_df(frame, Reader(rating_scale=scale)).build_full_trainset()


def ranking_candidates(
    held_out: pd.DataFrame, seen: pd.DataFrame, n_negatives: int, seed: int
) -> pd.DataFrame:
    """Rows to score for ranking: each user's held-out items plus sampled negatives.

    Negatives are items the user has no interaction with in any split (`seen`).
    Returns columns user, item, label, where label is 1 for held-out items.
    """
    rng = np.random.default_rng(seed)
    catalog = seen["item"].drop_duplicates().sort_values().tolist()
    seen_by_user = user_items(seen)

    rows: list[tuple[str, str, int]] = []
    for user, positives in user_items(held_out).items():
        excluded = seen_by_user[user]
        rows.extend((user, item, 1) for item in sorted(positives))
        unseen_count = len(catalog) - len(excluded)
        if unseen_count <= n_negatives:
            negatives = {item for item in catalog if item not in excluded}
        else:
            negatives: set[str] = set()
            while len(negatives) < n_negatives:
                item = catalog[int(rng.integers(len(catalog)))]
                if item not in excluded:
                    negatives.add(item)
        rows.extend((user, item, 0) for item in sorted(negatives))
    return pd.DataFrame(rows, columns=["user", "item", "label"])
