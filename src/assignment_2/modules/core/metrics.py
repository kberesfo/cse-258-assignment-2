"""Pure metric functions: arrays in, numbers out."""

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata


def calc_mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def calc_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of predictions that are correct."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def calc_ber(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Balanced error rate."""
    p = np.asarray(y_pred, dtype=bool)
    l = np.asarray(y_true, dtype=bool)

    TP = np.sum(p & l)
    FP = np.sum(p & ~l)
    TN = np.sum(~p & ~l)
    FN = np.sum(~p & l)

    if (TP + FN) == 0 or (TN + FP) == 0:
        raise ValueError("BER undefined: y_true contains only one class")

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return float(1 - 1 / 2 * (TPR + TNR))


def calc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve; tied scores count as half."""
    labels = np.asarray(y_true, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: y_true contains only one class")
    ranks = rankdata(np.asarray(y_score, dtype=float))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_share(counts: pd.Series, frac: float) -> float:
    """Share of all rows held by the top `frac` of items (e.g. 0.01 for the top 1%)."""
    counts = counts.sort_values(ascending=False)
    k = max(1, math.ceil(len(counts) * frac))
    return float(counts.iloc[:k].sum() / counts.sum())


def ranking_per_user(
    users: ArrayLike, labels: ArrayLike, scores: ArrayLike, k: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Hit@k, NDCG@k, and MRR for each user, with ties broken at random.

    Every user must have at least one positive label. Returns one array per metric, so
    the overall score is the mean and callers can resample users.
    """
    codes, _ = pd.factorize(np.asarray(users))
    labels, scores = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    tiebreak = np.random.default_rng(seed).random(len(codes))
    order = np.lexsort((tiebreak, -scores, codes))
    codes, labels = codes[order], labels[order]
    n_users = int(codes.max()) + 1

    ranks = pd.Series(codes).groupby(codes).cumcount().to_numpy() + 1
    in_top_k = ranks <= k
    positive = labels > 0

    hits = np.bincount(codes[positive & in_top_k], minlength=n_users) > 0
    dcg = np.bincount(codes, weights=positive * in_top_k / np.log2(ranks + 1), minlength=n_users)
    n_positive = np.bincount(codes[positive], minlength=n_users)
    if (n_positive == 0).any():
        raise ValueError("every user needs at least one positive label")
    ideal = np.cumsum(np.r_[0, 1 / np.log2(np.arange(2, k + 2))])[np.minimum(n_positive, k)]

    first_hit = np.full(n_users, len(codes) + 1)
    np.minimum.at(first_hit, codes[positive], ranks[positive])

    return {f"hit@{k}": hits.astype(float), f"ndcg@{k}": dcg / ideal, "mrr": 1 / first_hit}
