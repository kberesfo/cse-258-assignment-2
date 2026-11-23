"""BPR-MF (Rendle et al. 2009) via `implicit`: learns to rank played games above others."""

from typing import Literal, Self

import implicit.gpu
import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from numpy.typing import NDArray

from assignment_2.modules.core import interaction_matrix
from assignment_2.modules.data.tasks import TrainData

from ..base import ScoringModel


class BPRModel(ScoringModel):
    """BPR-MF: score(u, i) = γ_u · γ_i + b_i, trained so positives (owned or played games)
    outrank sampled ones."""

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        positives: Literal["played", "owned"] = "played",
        min_minutes: int = 120,
        use_gpu: bool = False,
        seed: int = 0,
    ) -> None:
        if use_gpu and not implicit.gpu.HAS_CUDA:
            raise RuntimeError("use_gpu=True, but this implicit build has no CUDA support")
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.positives = positives
        self.min_minutes = min_minutes
        self.seed = seed
        self.use_gpu = use_gpu

        # set by fit()
        self.model = None
        self.users: pd.Index | None = None  # user_id -> row of user_factors
        self.items: pd.Index | None = None  # item_id -> row of item_factors

    def fit(self, data: TrainData) -> Self:
        rows = data.interactions
        if self.positives == "played":
            rows = rows[rows.playtime_forever >= self.min_minutes]  # games played ≥ 2 h
        # "owned": every row in play_train, including games under min_minutes
        X, self.users, self.items = interaction_matrix(rows)

        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            random_state=self.seed,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            num_threads=1,  # CPU: one thread so runs repeat exactly (ignored on GPU)
        )

        self.model.fit(X.astype(np.float32), show_progress=False)
        if self.use_gpu:
            self.model = self.model.to_cpu()  # factors back to NumPy for predict()

        return self

    def predict(self, rows: pd.DataFrame) -> NDArray[np.float64]:
        if self.model is None:
            raise RuntimeError("BPRModel.predict called before fit")
        u = self.users.get_indexer(rows.user_id)  # -1 = no positives in training
        i = self.items.get_indexer(rows.item_id)  # -1 = nobody's positive in training
        ok = (u >= 0) & (i >= 0)

        scores = np.zeros(len(rows))
        # item_factors has one extra column (the item bias), so this includes b_i
        scores[ok] = np.einsum(
            "nk,nk->n", self.model.user_factors[u[ok]], self.model.item_factors[i[ok]]
        )
        return scores
