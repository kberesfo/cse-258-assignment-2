"""prep stage for Steam: sample users, split, and cache the tables as parquet.

    artifacts/steam/<tag>/              tag = "users_0.1", "full", ...
        interactions_{train,valid,test}.parquet
        reviews_{train,valid,test}.parquet
        items.parquet, bundles.parquet

Later stages and notebooks read these with load_prepped(), which takes seconds instead
of re-parsing the raw files. Changing [sample] or [split] in config.toml writes a new
folder (or overwrites the same tag), so runs at different sizes never mix.
"""

from pathlib import Path

import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core.split import Splits, split_by_user
from assignment_2.modules.core.steam import load_steam

SPLITS = ("train", "valid", "test")


def prepped_dir(cfg: Config) -> Path:
    return cfg.paths.art_dir / "steam" / cfg.sample.tag


def prep(cfg: Config) -> Path:
    out = prepped_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    data = load_steam(cfg.paths.data_dir / "steam", user_frac=cfg.sample.user_frac, seed=cfg.seed)
    print(f"[prep] {data.summary()}")

    tables: dict[str, Splits] = {
        "interactions": split_by_user(data.interactions, cfg.split, cfg.seed),
        "reviews": split_by_user(data.reviews, cfg.split, cfg.seed, time_col="posted"),
    }
    for name, s in tables.items():
        print(f"[prep] {name}: {s.summary()}")
        for split in SPLITS:
            getattr(s, split).to_parquet(out / f"{name}_{split}.parquet", index=False)

    data.items.to_parquet(out / "items.parquet", index=False)
    data.bundles.to_parquet(out / "bundles.parquet", index=False)
    print(f"[prep] wrote {out}")
    return out


def load_prepped(cfg: Config) -> dict[str, pd.DataFrame]:
    """{'interactions_train': df, ..., 'items': df, 'bundles': df} for the configured sample."""
    out = prepped_dir(cfg)
    if not out.is_dir():
        raise FileNotFoundError(f"{out} not found; run `make prep` (or `uv run assignment-2 prep`)")
    return {p.stem: pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))}
