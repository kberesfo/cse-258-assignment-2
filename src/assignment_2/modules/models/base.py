"""The contract between models and the evaluation code.

A model is a function `fit_<name>(cfg)` that trains on the train split and calls
`save_predictions` with a `predict(table) -> scores` function. What each score means
depends on the task:

    rating     the predicted rating
    binary     the probability that label = 1
    ranking    any score where higher means a better match for the user
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core.data import eval_table


class SkipModel(Exception):
    """Raised by a model that doesn't apply to the configured task."""


def save_predictions(cfg: Config, name: str, predict: Callable[[pd.DataFrame], np.ndarray]) -> None:
    """Score the valid and test rows, then save one array per split."""
    for split in ("valid", "test"):
        table = eval_table(cfg, split)
        scores = np.asarray(predict(table), dtype=float)
        if scores.shape != (len(table),):
            raise ValueError(f"{name}: got shape {scores.shape} for {len(table)} rows on {split}")
        if not np.isfinite(scores).all():
            raise ValueError(f"{name}: predictions on {split} contain NaN or infinity")
        np.save(cfg.paths.preds / f"{name}.{split}.npy", scores)
