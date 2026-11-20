# base.py
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from surprise import AlgoBase, Dataset, Reader, Trainset


class RatingModel(ABC):
    """Anything that fits on a ratings DataFrame and predicts ratings for (user, item) pairs."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Fit on rows with user / item / rating columns; return self so calls can chain."""

    @abstractmethod
    def predict(self, rows: pd.DataFrame) -> NDArray:
        """Predicted ratings for aligned users and items (unseen IDs get the model's fallback)."""


class SurpriseModel(RatingModel):
    """Shared plumbing for Surprise algorithms. Subclasses only say which algorithm to build."""

    def __init__(self, scale: tuple[float, float] = (1, 5)) -> None:
        self.scale = scale
        self.algo: AlgoBase | None = None

    @abstractmethod
    def _make_algo(self) -> AlgoBase:
        """A fresh, unfitted Surprise algorithm with this model's hyperparameters."""

    def fit(self, train: pd.DataFrame) -> Self:
        self.algo = self._make_algo()
        self.algo.fit(self._trainset(train))

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray:
        if self.algo is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict called before fit")

        return np.array([
            self.algo.predict(u, i).est
            for u, i in zip(rows["user"], rows["item"], strict=True)
        ])

    def _trainset(self, df: pd.DataFrame) -> Trainset:
        reader = Reader(rating_scale=self.scale)

        return Dataset.load_from_df(
            df[["user", "item", "rating"]], reader
        ).build_full_trainset()
