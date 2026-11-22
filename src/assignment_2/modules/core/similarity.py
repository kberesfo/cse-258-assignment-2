from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

SimMetric = Literal["jaccard", "cosine", "overlap"]


def column_similarity(X: sp.csr_matrix, metric: SimMetric = "jaccard") -> sp.csr_matrix:
    """Pairwise similarity between the columns of a 0/1 matrix.

    Column j is treated as the set of rows where it's nonzero; the result has the same values
    as jaccard / cosine / overlap on each pair of those sets, for every pair at once. Pairs
    with no rows in common stay 0 (not stored), so the result is sparse. Diagonal is 0.
    """
    X = (X > 0).astype(np.float64)  # 0/1, whatever X held
    inter = X.T @ X
    n = np.asarray(X.sum(axis=0)).ravel()
    c = inter.tocoo()
    row, col, both = c.row, c.col, c.data
    n_i, n_j = n[row], n[col]
    match metric:
        case "jaccard":
            denom = n_i + n_j - both
        case "cosine":
            denom = np.sqrt(n_i * n_j)
        case "overlap":
            denom = np.minimum(n_i, n_j)
        case _:
            raise ValueError(f"unknown metric {metric!r}")

    S = sp.csr_matrix((both / denom, (row, col)), shape=inter.shape)
    S.setdiag(0)
    S.eliminate_zeros()

    return S


def interaction_matrix(df: pd.DataFrame) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    """0/1 user × item matrix from (user_id, item_id) rows, plus the id → row/col index maps."""
    users = pd.Index(df.user_id.unique())
    items = pd.Index(df.item_id.unique())
    X = sp.csr_matrix(
        (np.ones(len(df)), (users.get_indexer(df.user_id), items.get_indexer(df.item_id))),
        shape=(len(users), len(items)),
    )
    return (X > 0).astype(np.float64), users, items  # > 0 collapses any duplicate rows


def item_similarity(X: sp.csr_matrix, metric: SimMetric = "jaccard"):
    """Game × game similarity from a user × game matrix: games compared by their users."""
    return column_similarity(X, metric)  # games, compared by their users


def user_similarity(X: sp.csr_matrix, metric: SimMetric = "jaccard"):
    """User × user similarity from a user × game matrix: users compared by their games."""
    return column_similarity(X.T.tocsr(), metric)
