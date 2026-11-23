from surprise import BaselineOnly

from ..surprise import SurpriseModel


class BiasModel(SurpriseModel):
    """f(u,i) = alpha + beta_u + beta_i, fit by ALS (Surprise BaselineOnly).

    Within one user only beta_i varies, so it ranks every user the same way:
    regularized popularity. A baseline, not a personalized model.
    """

    def __init__(
        self,
        reg_u: float = 15,
        reg_i: float = 10,
        n_epochs: int = 10,
        target: str = "playtime",
        neg_ratio: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__(target=target, neg_ratio=neg_ratio, seed=seed)
        self.reg_u, self.reg_i, self.n_epochs = reg_u, reg_i, n_epochs

    def _make_algo(self) -> BaselineOnly:
        options = {"method": "als", "reg_u": self.reg_u, "reg_i": self.reg_i,
                   "n_epochs": self.n_epochs}  # fmt: skip
        return BaselineOnly(bsl_options=options, verbose=False)

    def _score(self, u, i):
        return self.trainset.global_mean + self.algo.bu[u] + self.algo.bi[i]
