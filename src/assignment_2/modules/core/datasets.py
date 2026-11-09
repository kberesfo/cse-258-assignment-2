"""Dataset loaders: one function per dataset, each returning the standard table.

The standard table has one row per interaction:

    user, item           required; converted to strings
    rating               required for task = "rating"
    label                required for task = "binary"; 0 or 1
    timestamp            optional; Unix seconds. Enables the temporal split
    anything else        kept, so models can build task-specific features from it

To add a dataset: write a loader, add it to LOADERS, and set `dataset` in config.toml.
Add a download entry in scripts/get_data if it needs one. `modcloth` is the template.
"""

import ast
import gzip
import json
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from assignment_2.modules.config import Config


def parse_record(line: str) -> dict:
    """Parse one line. Many course datasets use Python dict literals rather than strict JSON."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    """Yield one record per line of a gzipped JSON-lines file."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data`")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield parse_record(line)


def load_synthetic(cfg: Config) -> pd.DataFrame:
    """Small fake dataset with user/item effects, for tests and running without a download."""
    rng = np.random.default_rng(cfg.seed)
    n_users, n_items, dim = 200, 100, 3

    user_bias, item_bias = rng.normal(
        0, 0.5, n_users), rng.normal(0, 0.5, n_items)
    user_factors, item_factors = (
        rng.normal(0, 0.5, (n_users, dim)),
        rng.normal(0, 0.5, (n_items, dim)),
    )
    # a long tail of item popularity
    popularity = 1 / np.arange(1, n_items + 1) ** 0.8
    popularity /= popularity.sum()

    rows = []
    for user in range(n_users):
        count = min(int(rng.lognormal(2.7, 0.6)) + 3, n_items)
        for item in rng.choice(n_items, count, replace=False, p=popularity):
            signal = user_bias[user] + item_bias[item] + \
                user_factors[user] @ item_factors[item]
            rating = float(
                np.clip(np.round(3.5 + signal + rng.normal(0, 0.5)), 1, 5))
            rows.append((f"u{user}", f"i{item}", rating, int(rating >= 4)))

    df = pd.DataFrame(rows, columns=["user", "item", "rating", "label"])
    start = 1_600_000_000  # September 2020
    df["timestamp"] = start + rng.integers(0, 2 * 365 * 86_400, len(df))
    return df


LOADERS: dict[str, Callable[[Config], pd.DataFrame]] = {
    "synthetic": load_synthetic,
}
