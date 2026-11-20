# bias.py (the class part)
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from surprise import AlgoBase, BaselineOnly, Dataset, Reader, Trainset

from .base import RatingModel


class BiasModel(RatingModel):
    """f(u,i) = alpha + beta_u + beta_i, fit by ALS (Surprise BaselineOnly)."""

    def __init__(
        self,
        reg_u: float,
        reg_i: float,
        n_epochs: int = 25,
        scale: tuple[float, float] = (1, 5)
    ) -> None:
        super().__init__(scale)
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.n_epochs = n_epochs
        self.model: AlgoBase | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        self.model = BaselineOnly(
            bsl_options={
                "method": "als",
                "reg_u": self.reg_u,
                "reg_i": self.reg_i,
                "n_epochs": self.n_epochs
            },
            verbose=False
        )

        self.model.fit(self._trainset(train))

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray:
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

    def _make_algo(self) -> BaselineOnly:
        return
