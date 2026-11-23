# models/registry.py
"""Every model the project can train, by name. Add new models here."""

import inspect
from typing import Any

from assignment_2.modules.config import Config

from .base import ScoringModel
from .ranking.bpr import BPRModel
from .ranking.knn import ItemKNNRanker, UserKNNRanker
from .rating.bias import BiasModel
from .rating.lfm import SVDModel

# ItemKNNRegressor, UserKNNRegressor	item_knn_reg, user_knn_reg


MODELS: dict[str, type[ScoringModel]] = {
    "bpr": BPRModel,
    "bias": BiasModel,
    "svd": SVDModel,
    "item_knn_rank": ItemKNNRanker,
    "user_knn_rank": UserKNNRanker,
}


def build(name: str, cfg: Config, **overrides: Any) -> ScoringModel:
    """Construct a model from its [models.<name>] settings, with `overrides` on top
    (plus cfg.seed and [split.play] min_minutes, if it takes them and they aren't set)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; options: {', '.join(MODELS)}")
    cls = MODELS[name]
    params = {**cfg.params(name), **overrides}
    accepts = inspect.signature(cls).parameters
    if "seed" in accepts:
        params.setdefault("seed", cfg.seed)
    if "min_minutes" in accepts:  # task A's "played" threshold
        params.setdefault("min_minutes", cfg.split.play.min_minutes)
    try:
        return cls(**params)
    except TypeError as e:  # a typo or stale key in config.toml
        raise TypeError(f"[models.{name}] in config doesn't match {cls.__name__}: {e}") from None
