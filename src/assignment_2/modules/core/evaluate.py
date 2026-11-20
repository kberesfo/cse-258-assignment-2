# evaluate.py: score a model's predictions on a candidate table, with uncertainty
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
    # Steps 1 and 3 from bootstrap_ci live here: make the generator from the seed,
    # then draw the (n_boot, n) index array. bootstrap_ci and compare both call this,
    # so the "same seed -> same resamples" guarantee is written once.
    raise NotImplementedError()


def bootstrap_ci(per_user: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    # (your steps 0–7, with 1 and 3 replaced by a call to _resample_indices)
    raise NotImplementedError()


def evaluate_scores(
    cand: pd.DataFrame, scores: np.ndarray, k: int, n_boot: int, seed: int
) -> tuple[dict[str, Metric], dict[str, np.ndarray]]:
    """Score one model on one candidate table.

    cand:   the fixed candidate table (user_id, item_id, label) from load_prepped.
    scores: model.predict(cand), one score per row, same order as cand.
    Returns (summary, per_user):
        summary:  {"ndcg@10": Metric(value, ci_low, ci_high), ...} for reporting
        per_user: {"ndcg@10": array of one value per user, ...} kept for compare()
    """
    # 1. Sanity check: len(scores) must equal len(cand). If not, the model scored the
    #    wrong rows. Raise instead of evaluating something meaningless.
    #    (Also worth checking: no NaN in scores. A NaN would silently sort to one end.)

    # 2. Turn row scores into per-user metrics with ranking_per_user, passing
    #    cand["user_id"], cand["label"], scores, k. Use the same seed for every model,
    #    so ties are broken the same way for everyone.
    #    -> dict like {"hit@10": array, "ndcg@10": array, "mrr": array}, one value per user.

    # 3. For each metric name and its per-user array:
    #      value = the array's mean (this is the number you'd report)
    #      low, high = bootstrap_ci(array, n_boot, seed)
    #      summary[name] = Metric(value, low, high)
    #    Use the same seed for every metric too; they're all resampled over the same users.

    # 4. Return (summary, per_user). per_user is what compare() needs. Without it,
    #    you'd have to recompute ranking_per_user to compare two models.
    raise NotImplementedError()


def compare(per_user_a: np.ndarray, per_user_b: np.ndarray, n_boot: int, seed: int) -> Metric:
    """Bootstrap CI of the *difference* (model a − baseline b) on the same resampled users.

    Why not just compare the two models' intervals? Most of the variation in a metric
    comes from *which users* are in the sample: easy users make every model look good.
    Looking at the difference on the same users cancels that out, so it can show a real
    improvement even when the two intervals overlap.

    Reading the result: ci_low > 0 means a beats b; ci_high < 0 means b beats a;
    an interval containing 0 means the data can't tell them apart.
    """
    # 1. Both arrays must describe the same users in the same order. Same length is the
    #    minimum check. (They will be, if both came from evaluate_scores on the same
    #    candidate table. Returning user IDs from ranking_per_user would let you check
    #    properly.)

    # 2. Point estimate: the difference of the two means (a's mean minus b's mean).

    # 3. Get the (n_boot, n) indices from _resample_indices. The SAME indices go to both
    #    arrays; that's the whole point.

    # 4. For each row of indices: mean of a at those indices minus mean of b at those
    #    indices. Vectorized: index both arrays with the full index array, take the mean
    #    along axis=1 for each, then subtract. -> n_boot simulated differences.

    # 5. 2.5th and 97.5th percentiles of those differences -> (low, high).

    # 6. Return Metric(value=difference, ci_low=low, ci_high=high).
    raise NotImplementedError()
