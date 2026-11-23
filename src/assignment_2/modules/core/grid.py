"""Grid search over model settings: one row per (setting, seed), with a progress bar.

Task-agnostic: the caller passes `score(params, seed) -> {metric: value}`, which builds,
fits and evaluates one model. `run_grid` loops over every combination of the grid's
values × seeds, times each fit, and returns (and optionally saves) the rows.
"""

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

Scorer = Callable[[dict[str, Any], int], Mapping[str, float]]


def expand(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the grid's values, as one dict per setting. {} -> [{}]."""
    for key, values in grid.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"grid[{key!r}] must be a list of values, got {values!r}")
        if not values:
            raise ValueError(f"grid[{key!r}] is empty")
    return [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())]


def run_grid(
    score: Scorer,
    grid: Mapping[str, Sequence[Any]],
    seeds: Sequence[int] = (0, 1, 2),
    base: Mapping[str, Any] | None = None,
    path: Path | None = None,
    desc: str | None = None,
) -> pd.DataFrame:
    """Score every setting in `grid` (on top of `base`) with every seed.

    Returns one row per (setting, seed): the full settings (base + grid), seed, the
    metrics `score` returned, and fit+score time in seconds. Saved to `path` as CSV
    if given, so a grid can be reloaded without rerunning it.
    """
    base = dict(base or {})
    runs = list(itertools.product(expand(grid), seeds))
    rows = []
    with tqdm(runs, desc=desc, unit="fit") as bar:
        for params, seed in bar:
            bar.set_postfix_str(", ".join(f"{k}={v}" for k, v in {**params, "seed": seed}.items()))
            start = time.perf_counter()
            metrics = dict(score({**base, **params}, seed))
            rows.append({**base, **params, "seed": seed, **metrics,
                         "seconds": round(time.perf_counter() - start, 1)})  # fmt: skip
    df = pd.DataFrame(rows)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def summarize(g: pd.DataFrame, by: str | list[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and std over seeds of each metric, per setting in `by`."""
    return g.groupby(by)[list(metrics)].agg(["mean", "std"]).round(4)
