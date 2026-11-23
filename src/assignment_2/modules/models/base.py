"""Shared model interface (fit / predict), plus the Surprise plumbing.

Every model: `fit(data) -> self`, then `predict(rows) -> scores` aligned to `rows`
(one float per row). What a score means depends on the task: for play, higher = more
likely to be played (only the order matters); for engagement, the predicted
log1p(playtime). `fit` receives a task's TrainData and takes what it needs; `predict`
receives rows keyed by user_id / item_id. Evaluation only uses this interface.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from assignment_2.modules.data.tasks import TrainData


class ScoringModel(ABC):
    @abstractmethod
    def fit(self, data: TrainData) -> Self: ...

    @abstractmethod
    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]: ...
