# models/ranking/registry.py
"""Every model the project can train, by name. Add new models here."""

import inspect

from assignment_2.modules.config import Config

from .base import ScoringModel
from .rating.bias import BiasModel
from .rating.lfm import SVDModel

# from .popularity import PopularityModel

MODELS: dict[str, type[ScoringModel]] = {
    # "popularity": PopularityModel,
    "bias": BiasModel,
    "svd": SVDModel,
}


def build(name: str, cfg: Config) -> ScoringModel:
    """Construct a model from its [models.<name>] settings (plus cfg.seed if it takes one)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; options: {', '.join(MODELS)}")
    cls = MODELS[name]
    params = cfg.params(name)
    if "seed" in inspect.signature(cls).parameters:
        params.setdefault("seed", cfg.seed)
    try:
        return cls(**params)
    except TypeError as e:  # a typo or stale key in config.toml
        raise TypeError(f"[models.{name}] in config doesn't match {cls.__name__}: {e}") from None
