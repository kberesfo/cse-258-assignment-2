# metrics.py: pure, per user
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import ranking_per_user


@dataclass(frozen=True)
class Metric:
    value: float
    ci_low: float | None
    ci_high: float | None


def bootstrap_ci(per_user: np.ndarray, n_boot: int,
                 seed: int) -> tuple[float, float]:
    pass


def evaluate_scores(cand: pd.DataFrame, scores: np.ndarray, k: int,
                    n_boot: int, seed: int) -> dict[str, Metric]:
    pass


def compare(per_user_a: np.ndarray, per_user_b: np.ndarray,
            n_boot: int, seed: int) -> Metric:
    """Bootstrap CI of the *difference* (model − baseline) on the same resampled users."""
    pass
