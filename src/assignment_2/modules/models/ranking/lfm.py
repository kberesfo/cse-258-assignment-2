# lfm.py (the class part)
from typing import Any, Self

import pandas as pd
from surprise import SVD, AlgoBase, Dataset, Reader, Trainset

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
        self.seed = seed
        self.svd_params = svd_params
        self.model: AlgoBase | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        self.model = SVD(random_state=self.seed, **self.svd_params)
        self.model.fit(self._trainset(train))
        return self

    def predict(self, rows):
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit"
            )

        return

    def _trainset(self, df: pd.DataFrame) -> Trainset:
        reader = Reader(rating_scale=self.scale)

        return Dataset.load_from_df(
            df[["user", "item", "rating"]], reader
        ).build_full_trainset()
