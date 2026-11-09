"""The evaluation protocol: every model is scored here, on the same rows.

Models write one score per row of `eval_table` to artifacts/preds/. This module reads
those files, so it doesn't depend on how any model works.
"""

import json
from collections.abc import Callable

import numpy as np

from assignment_2.modules.config import Config
from assignment_2.modules.core.data import eval_table, target_column
from assignment_2.modules.core.metrics import (
    calc_accuracy,
    calc_auc,
    calc_ber,
    calc_mae,
    calc_mse,
    ranking_per_user,
)

# Binary models output probabilities, which accuracy and BER threshold at 0.5.
TASK_METRICS: dict[str, dict[str, Callable]] = {
    "rating": {"mse": calc_mse, "mae": calc_mae},
    "binary": {
        "auc": calc_auc,
        "accuracy": lambda y, s: calc_accuracy(y, s >= 0.5),
        "ber": lambda y, s: calc_ber(y, s >= 0.5),
    },
}


def _statistics(cfg: Config, table, scores: np.ndarray) -> tuple[int, dict[str, Callable]]:
    """Return the sample count and, per metric, a function of resampled indices.

    Ranking resamples users, because a user's candidates aren't independent rows.
    """
    if cfg.task == "ranking":
        per_user = ranking_per_user(
            table["user"], table["label"], scores, cfg.evaluation.k, seed=cfg.seed
        )
        n = len(next(iter(per_user.values())))
        return n, {name: (lambda i, v=v: float(v[i].mean())) for name, v in per_user.items()}

    y = table[target_column(cfg)].to_numpy(dtype=float)
    return len(y), {
        name: (lambda i, f=f: f(y[i], scores[i])) for name, f in TASK_METRICS[cfg.task].items()
    }


def score_model(cfg: Config, table, scores: np.ndarray) -> dict[str, dict]:
    """Each metric's value and bootstrap 95% interval.

    The seed is fixed, so every model is scored on the same resamples.
    """
    n, statistics = _statistics(cfg, table, scores)
    point = {name: stat(np.arange(n)) for name, stat in statistics.items()}

    rng = np.random.default_rng(cfg.seed)
    resampled: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(cfg.evaluation.n_bootstrap):
        idx = rng.integers(0, n, n)
        for name, stat in statistics.items():
            try:
                resampled[name].append(stat(idx))
            except ValueError:  # a resample can contain only one class
                resampled[name].append(np.nan)

    results = {}
    for name, value in point.items():
        low, high = (
            (float(x) for x in np.nanpercentile(resampled[name], [2.5, 97.5]))
            if resampled[name]
            else (None, None)
        )
        results[name] = {"value": value, "ci_low": low, "ci_high": high}
    return results


def evaluate(cfg: Config, split: str = "valid") -> None:
    """Score each model's saved predictions and record them in cfg.paths.metrics.

    Results are stored per split, so evaluating valid doesn't overwrite test.
    """
    table = eval_table(cfg, split)
    suffix = f".{split}.npy"
    files = sorted(cfg.paths.preds.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(
            f"no predictions in {cfg.paths.preds}; run `make train`")

    models = {}
    for path in files:
        scores = np.load(path)
        if len(scores) != len(table):
            raise ValueError(
                f"{path.name} has {len(scores)} rows, expected {len(table)}")
        name = path.name.removesuffix(suffix)
        models[name] = score_model(cfg, table, scores)
        line = "  ".join(f"{m}={r['value']:.4f}" for m,
                         r in models[name].items())
        print(f"  {name:20s} {line}")

    path = cfg.paths.metrics
    results = json.loads(path.read_text()) if path.is_file() else {}
    results[split] = {"dataset": cfg.dataset,
                      "task": cfg.task, "models": models}
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
