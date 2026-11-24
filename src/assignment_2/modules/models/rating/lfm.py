from typing import Any

import numpy as np
from surprise import SVD

from ..surprise import SurpriseModel, Target


class SVDModel(SurpriseModel):
    """Funk-style LFM: f(u,i) = alpha + beta_u + beta_i + gamma_u . gamma_i, fit by SGD."""

    def __init__(
        self,
        target: Target = "minutes",
        neg_ratio: int = 0,
        seed: int = 0,
        **svd_params: Any,  # n_factors, n_epochs, lr_all, reg_all, biased, ...
    ) -> None:
        super().__init__(target=target, neg_ratio=neg_ratio, seed=seed)
        SVD(**svd_params)  # fail now on a typo or unknown setting, not at fit time
        self.svd_params = svd_params

    def _make_algo(self) -> SVD:
        return SVD(random_state=self.seed, **self.svd_params)

    def _score(self, u, i):
        a = self.algo
        dot = np.einsum("nk,nk->n", a.pu[u], a.qi[i])
        return self.trainset.global_mean + a.bu[u] + a.bi[i] + dot if a.biased else dot
