"""Score a model's predictions with bootstrap uncertainty, and compare two models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import ranking_per_user


@dataclass(frozen=True)
class Metric:
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
    """95% bootstrap interval for the mean of per_user (one value per user)."""

    n = len(per_user)
    idx = _resample_indices(n, n_boot, seed)
    users_value = per_user[idx]
    means = np.mean(users_value, axis=1)
    low, high = np.percentile(means, [2.5, 97.5])

    return float(low), float(high)


def evaluate_ranking(
    cand: pd.DataFrame, scores: np.ndarray, k: int, n_boot: int, seed: int
) -> tuple[dict[str, Metric], dict[str, pd.Series]]:
    """Score one model on one candidate table (task A).

    cand:   the fixed candidate table (user_id, item_id, label) from load_prepped.
    scores: model.predict(cand), one score per row, same order as cand.
    Returns (summary, per_user):
        summary:  {"ndcg@10": Metric(value, ci_low, ci_high), ...} for reporting
        per_user: {"ndcg@10": Series of one value per user, indexed by user_id, ...} kept for compare()
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(cand):
        raise ValueError(
            f"got {len(scores):,} scores for {len(cand):,} candidate rows; "
            "predict() must return one score per row, in order"
        )
    if np.isnan(scores).any():
        raise ValueError(f"{int(np.isnan(scores).sum()):,} scores are NaN")

    # Same seed for every model, so ties are broken identically for everyone.
    arrays = ranking_per_user(cand["user_id"], cand["label"], scores, k=k, seed=seed)
    # Label each value with its user, so compare() can check both sides cover the same
    # users in the same order. ranking_per_user numbers users in order of first
    # appearance, which is the order pd.unique returns.
    users = pd.unique(cand["user_id"])
    per_user = {name: pd.Series(v, index=users, name=name) for name, v in arrays.items()}

    # Same seed for every metric: each is resampled over the same simulated users.
    summary: dict[str, Metric] = {}
    for name, values in per_user.items():
        low, high = bootstrap_ci(values.to_numpy(), n_boot, seed)
        summary[name] = Metric(value=float(values.mean()), ci_low=low, ci_high=high)
    return summary, per_user


def compare(a: pd.Series, b: pd.Series, n_boot: int, seed: int) -> Metric:
    """Bootstrap CI of the *difference* (model a − baseline b) on the same resampled users.

    Why not just compare the two models' intervals? Most of the variation in a metric
    comes from *which users* are in the sample: easy users make every model look good.
    Looking at the difference on the same users cancels that out, so it can show a real
    improvement even when the two intervals overlap.

    Reading the result: ci_low > 0 means a beats b; ci_high < 0 means b beats a;
    an interval containing 0 means the data can't tell them apart.
    """
    if not a.index.equals(b.index):
        raise ValueError("a and b must be per-user results for the same users, in the same order")

    # 2. Point estimate: the difference of the two means (a's mean minus b's mean).
    diff = a.mean() - b.mean()
    # 3. Get the (n_boot, n) indices from _resample_indices. The SAME indices go to both
    #    arrays; that's the whole point.
    idx = _resample_indices(len(a), n_boot, seed)  # (n_boot, n), shared by both
    a_np, b_np = a.to_numpy(), b.to_numpy()
    diffs = a_np[idx].mean(axis=1) - b_np[idx].mean(axis=1)  # n_boot simulated differences
    low, high = np.percentile(diffs, [2.5, 97.5])

    return Metric(value=float(diff), ci_low=float(low), ci_high=float(high))
