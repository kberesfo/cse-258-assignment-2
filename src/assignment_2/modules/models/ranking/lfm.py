# lfm.py (the class part)
from typing import Any, Self

import pandas as pd
from surprise import SVD

from .base import RatingModel


class SVDModel(RatingModel):
    """Funk-style LFM: f(u,i) = alpha + beta_u + beta_i + gamma_u . gamma_i, fit by SGD."""

    def __init__(
            self,
            scale: tuple[float, float] = (1, 5),
            seed: int = 0,
            **svd_params: Any
    ) -> None:
        super().__init__(scale)
        self.seed, self.svd_params = seed, svd_params

    def fit(self, train: pd.DataFrame) -> Self:

        return self

    def _make_algo(self) -> SVD:
        return SVD(random_state=self.seed, **self.svd_params)
