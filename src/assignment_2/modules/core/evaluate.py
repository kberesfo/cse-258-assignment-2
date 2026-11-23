"""Score a model's predictions with bootstrap uncertainty, and compare two models.

Both tasks return the same shape, (summary, per_user), so compare() works for either:
    summary:  {"hit@10": Metric(value, ci_low, ci_high), ...}       for reporting
    per_user: {"hit@10": Series, one value per user, by user_id}    for compare()
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import ranking_per_user, within_user_spearman


@dataclass(frozen=True)
class Metric:
    """A metric's value over users, with its 95% bootstrap interval (None if not computed)."""

    value: float
    ci_low: float | None
    ci_high: float | None


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) user indices drawn with replacement. Shared by bootstrap_ci and compare.

    Same (n, n_boot, seed) -> same array, which is what makes every model's interval,
    and every difference, use identical simulated user populations.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci(per_user: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% bootstrap interval for the mean of per_user (one value per user).

    NaN values (users where a metric is undefined) are skipped in each resample's mean.
    """
    idx = _resample_indices(len(per_user), n_boot, seed)
    means = np.nanmean(per_user[idx], axis=1)  # one mean per simulated sample
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summarize(per_user: dict[str, pd.Series], n_boot: int, seed: int) -> dict[str, Metric]:
    # Same seed for every metric: each is resampled over the same simulated users.
    summary: dict[str, Metric] = {}
    for name, values in per_user.items():
        if values.isna().all():  # e.g. Spearman of a model that predicts one value per user
            summary[name] = Metric(value=float("nan"), ci_low=None, ci_high=None)
            continue
        low, high = bootstrap_ci(values.to_numpy(), n_boot, seed)
        summary[name] = Metric(value=float(values.mean()), ci_low=low, ci_high=high)
    return summary


def _check_predictions(pred: np.ndarray, n_rows: int, what: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if len(pred) != n_rows:
        raise ValueError(
            f"got {len(pred):,} predictions for {n_rows:,} {what} rows; "
            "predict() must return one value per row, in order"
        )
    if np.isnan(pred).any():
        raise ValueError(f"{int(np.isnan(pred).sum()):,} predictions are NaN")
    return pred


def evaluate_ranking(
    cand: pd.DataFrame, scores: np.ndarray, k: int, n_boot: int, seed: int
) -> tuple[dict[str, Metric], dict[str, pd.Series]]:
    """Score one model on one candidate table (task A): Hit@k, NDCG@k, MRR per user.

    cand:   the fixed candidate table (user_id, item_id, label) from load_prepped.
    scores: model.predict(cand), one score per row, same order as cand.
    """
    scores = _check_predictions(scores, len(cand), "candidate")

    # Same seed for every model, so ties are broken identically for everyone.
    arrays = ranking_per_user(cand["user_id"], cand["label"], scores, k=k, seed=seed)
    # Label each value with its user, so compare() can check both sides cover the same
    # users in the same order. ranking_per_user numbers users in order of first
    # appearance, which is the order pd.unique returns.
    users = pd.unique(cand["user_id"])
    per_user = {name: pd.Series(v, index=users, name=name) for name, v in arrays.items()}
    return _summarize(per_user, n_boot, seed), per_user


def evaluate_engagement(
    rows: pd.DataFrame, pred: np.ndarray, n_boot: int, seed: int
) -> tuple[dict[str, Metric], dict[str, pd.Series]]:
    """Score one model's playtime predictions on held-out owned games (task B).

    rows: own_valid or own_test (user_id, item_id, playtime_forever).
    pred: model.predict(rows), predicted log1p(minutes), one per row, same order.
    Metrics are computed per user, then averaged, so every user counts equally:
        mse, mae:  mean squared / absolute error over the user's held-out games
        spearman:  within-user rank correlation; NaN (skipped) where undefined
    """
    pred = _check_predictions(pred, len(rows), "held-out")
    y = np.log1p(rows["playtime_forever"].to_numpy(float))
    users = rows["user_id"].to_numpy()

    err = pred - y
    mse = pd.Series(err**2).groupby(users).mean()
    per_user = {
        "mse": mse.rename("mse"),
        "mae": pd.Series(np.abs(err)).groupby(users).mean().rename("mae"),
        "spearman": within_user_spearman(users, y, pred).reindex(mse.index).rename("spearman"),
    }
    return _summarize(per_user, n_boot, seed), per_user


def compare(a: pd.Series, b: pd.Series, n_boot: int, seed: int) -> Metric:
    """Bootstrap CI of the *difference* (model a − baseline b) on the same resampled users.

    Why not just compare the two models' intervals? Most of the variation in a metric
    comes from *which users* are in the sample: easy users make every model look good.
    Looking at the difference on the same users cancels that out, so it can show a real
    improvement even when the two intervals overlap.

    Users where either model's value is NaN (an undefined Spearman) are left out, so the
    pairing stays user by user.

    Reading the result: ci_low > 0 means a is higher than b; ci_high < 0 means b is higher;
    an interval containing 0 means the data can't tell them apart. (For an error such as
    MSE, lower is better, so a beats b when ci_high < 0.)
    """
    if not a.index.equals(b.index):
        raise ValueError("a and b must be per-user results for the same users, in the same order")
    both = a.notna() & b.notna()
    if not both.any():
        raise ValueError("no users where both a and b are defined")
    a_np, b_np = a[both].to_numpy(), b[both].to_numpy()

    idx = _resample_indices(len(a_np), n_boot, seed)  # (n_boot, n), shared by both
    diffs = a_np[idx].mean(axis=1) - b_np[idx].mean(axis=1)  # n_boot simulated differences
    low, high = np.percentile(diffs, [2.5, 97.5])
    return Metric(value=float(a_np.mean() - b_np.mean()), ci_low=float(low), ci_high=float(high))
