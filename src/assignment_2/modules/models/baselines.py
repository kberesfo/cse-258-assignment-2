"""Trivial and course baselines. They use only the standard table, so they fit any dataset."""

import numpy as np
import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core.data import load_split, target_column
from assignment_2.modules.core.features import IdIndex
from assignment_2.modules.models.base import SkipModel, save_predictions


def fit_trivial(cfg: Config) -> None:
    """Train mean for rating/binary tasks; random scores for ranking."""
    if cfg.task == "ranking":
        rng = np.random.default_rng(cfg.seed)
        save_predictions(cfg, "trivial", lambda table: rng.random(len(table)))
        return

    mean = load_split(cfg, "train")[target_column(cfg)].mean()
    save_predictions(cfg, "trivial", lambda table: np.full(len(table), mean))


def fit_popularity(cfg: Config) -> None:
    """Item effect only: smoothed item mean, or interaction count for ranking."""
    train = load_split(cfg, "train")
    items = train.groupby("item")

    if cfg.task == "ranking":
        counts = items.size()

        def predict(table: pd.DataFrame) -> np.ndarray:
            return table["item"].map(counts).fillna(0).to_numpy()
    else:
        strength = cfg.params("popularity").get("strength", 5.0)
        target = target_column(cfg)
        mean = train[target].mean()
        grouped = items[target]
        smoothed = (grouped.sum() + strength * mean) / (grouped.size() + strength)

        def predict(table: pd.DataFrame) -> np.ndarray:
            return table["item"].map(smoothed).fillna(mean).to_numpy()

    save_predictions(cfg, "popularity", predict)


def fit_bias(cfg: Config) -> None:
    """Rating = alpha + beta_user + beta_item, fit by regularized coordinate descent."""
    if cfg.task == "ranking":
        raise SkipModel("user bias is constant within a user, so this reduces to popularity")

    params = {"lam": 1.0, "iters": 20} | cfg.params("bias")
    lam = params["lam"]
    train = load_split(cfg, "train")
    index = IdIndex.from_frame(train)
    users, items = index.user_codes(train["user"]), index.item_codes(train["item"])
    y = train[target_column(cfg)].to_numpy(dtype=float)

    user_counts = np.bincount(users, minlength=index.n_users)
    item_counts = np.bincount(items, minlength=index.n_items)
    alpha, beta_user, beta_item = y.mean(), np.zeros(index.n_users), np.zeros(index.n_items)
    for _ in range(params["iters"]):
        residual = y - alpha - beta_item[items]
        beta_user = np.bincount(users, residual, index.n_users) / (lam + user_counts)
        residual = y - alpha - beta_user[users]
        beta_item = np.bincount(items, residual, index.n_items) / (lam + item_counts)
        alpha = float(np.mean(y - beta_user[users] - beta_item[items]))

    # A trailing zero means unknown IDs, which map to -1, get no adjustment.
    beta_user, beta_item = np.append(beta_user, 0.0), np.append(beta_item, 0.0)

    def predict(table: pd.DataFrame) -> np.ndarray:
        u, i = index.user_codes(table["user"]), index.item_codes(table["item"])
        scores = alpha + beta_user[u] + beta_item[i]
        return np.clip(scores, 0, 1) if cfg.task == "binary" else scores

    save_predictions(cfg, "bias", predict)
