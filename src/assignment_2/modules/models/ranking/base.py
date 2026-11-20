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

