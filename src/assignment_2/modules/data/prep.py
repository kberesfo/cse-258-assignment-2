"""The prep stage: sample users, split, and cache the tables as parquet.

    artifacts/steam/<tag>/              tag = "users_0.1", "full", ...
        play_{train,valid,test}.parquet         task A: held-out games played >= min_minutes
        play_{valid,test}_candidates.parquet    fixed ranking candidates for task A
        own_{train,valid,test}.parquet          task B: held-out owned games (log playtime)
        reviews_{train,valid,test}.parquet
        items.parquet, bundles.parquet, meta.json

Later stages and notebooks read these with load_prepped(), which takes seconds instead
of re-parsing the raw files. load_prepped() refuses a cache whose meta.json settings
don't match the current config, so changing the seed or [split] without re-running
prep fails loudly instead of silently reusing old splits.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.config.split import Split

from .candidates import ranking_candidates
from .split import Splits, split_by_user
from .steam import load_steam

SPLITS = ("train", "valid", "test")
META = "meta.json"


def prepped_dir(cfg: Config) -> Path:
    return cfg.paths.art_dir / "steam" / cfg.sample.tag


def _settings(cfg: Config) -> dict[str, Any]:
    """Everything that changes the cached tables. Compared on load."""
    return {
        "dataset": "steam",
        "seed": cfg.seed,
        "sample": cfg.sample.model_dump(mode="json"),
        "split": cfg.split.model_dump(mode="json"),
        "evaluation": cfg.evaluation.model_dump(mode="json"),
    }


def _table_stats(df: pd.DataFrame) -> dict[str, Any]:
    users, items = df["user_id"].nunique(), df["item_id"].nunique()
    return {
        "rows": len(df),
        "users": users,
        "items": items,
        # Share of the user x item matrix that is empty.
        "sparsity": 1 - len(df) / (users * items) if users and items else None,
    }


def _split_stats(s: Splits) -> dict[str, Any]:
    return {
        **{name: _table_stats(getattr(s, name)) for name in SPLITS},
        "cold_dropped": s.n_cold_dropped,
    }


def _check_nonempty(name: str, s: Splits, split_cfg: Split) -> None:
    """Raise if a split that should hold rows came out empty.

    Count mode always holds out rows, so valid and test must both be non-empty.
    Frac mode only checks a split whose fraction is > 0.
    """
    expected = [
        split for split in ("valid", "test")
        if split_cfg.holdout == "count" or getattr(split_cfg, f"{split}_frac") > 0
    ]  # fmt: skip
    empty = [split for split in expected if getattr(s, split).empty]
    if empty:
        raise ValueError(
            f"{name}: {' and '.join(empty)} split is empty; "
            "check the [split] and [sample] settings"
        )


def prep(cfg: Config) -> Path:
    out = prepped_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    data = load_steam(cfg.paths.data_dir / "steam",
                      user_frac=cfg.sample.user_frac, seed=cfg.seed)
    print(f"[prep] {data.summary()}")

    owned = data.interactions  # every owned (user, game), with playtime
    is_played = owned.playtime_forever >= cfg.split.play.min_minutes
    played, unplayed = owned[is_played], owned[~is_played]

    # Task A: hold out played games only.
    play = split_by_user(played, cfg.split.play, cfg.seed)
    # Owned-but-unplayed games go into train only: they show what the user owns, but are
    # never held out, since we can't say whether the user would have played them.
    # Kept only for users already in play.train, so that users the split dropped don't
    # come back with only unplayed games.
    extra = unplayed[unplayed.user_id.isin(play.train.user_id)]
    play = replace(play, train=pd.concat(
        [play.train, extra], ignore_index=True))

    # Task B: hold out any owned game; the target is its playtime.
    own = split_by_user(owned, cfg.split.engagement, cfg.seed)
    reviews = split_by_user(
        data.reviews, cfg.split.reviews, cfg.seed, time_col="posted")

    tables = {"play": play, "own": own, "reviews": reviews}
    split_cfgs = {"play": cfg.split.play,
                  "own": cfg.split.engagement, "reviews": cfg.split.reviews}
    for name, s in tables.items():
        print(f"[prep] {name}: {s.summary()}")
        _check_nonempty(name, s, split_cfgs[name])

    # Fixed ranking candidates for task A: sampled once here so every model ranks the same games.
    candidates = {
        split: ranking_candidates(
            held_out=getattr(play, split),
            train=play.train,  # includes unplayed: negatives weighted by ownership popularity
            owned=owned,  # every game each user owns, so unplayed games are never negatives
            n_negatives=cfg.evaluation.n_negatives,
            seed=cfg.seed + offset,
            negatives=cfg.evaluation.negatives,
        )
        for offset, split in enumerate(("valid", "test"), start=1)
    }
    for split, c in candidates.items():
        print(
            f"[prep] {split} candidates: {len(c):,} rows, {c.user_id.nunique():,} users")

    # Write meta.json last, after every table: a cache without it is incomplete.
    (out / META).unlink(missing_ok=True)
    for name, s in tables.items():
        for split in SPLITS:
            getattr(s, split).to_parquet(
                out / f"{name}_{split}.parquet", index=False)
    for split, c in candidates.items():
        c.to_parquet(out / f"play_{split}_candidates.parquet", index=False)
    data.items.to_parquet(out / "items.parquet", index=False)
    data.bundles.to_parquet(out / "bundles.parquet", index=False)

    meta = {
        "settings": _settings(cfg),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "stats": {
            "play": {
                "min_minutes": cfg.split.play.min_minutes,
                "played_all": _table_stats(played),
                "unplayed_added_to_train": len(extra),
                **_split_stats(tables["play"]),
            },
            "own": {"all": _table_stats(owned), **_split_stats(tables["own"])},
            "reviews": {
                "all": _table_stats(data.reviews),
                "undated_share": float(data.reviews["posted"].isna().mean()),
                **_split_stats(tables["reviews"]),
            },
            "candidates": {
                split: {
                    "rows": len(c),
                    "users": int(c["user_id"].nunique()),
                    "positives": int(c["label"].sum()),
                    "negatives_per_user": cfg.evaluation.n_negatives,
                }
                for split, c in candidates.items()
            },
            "items": {"rows": len(data.items)},
            "bundles": {
                "bundles": int(data.bundles["bundle_id"].nunique()),
                "rows": len(data.bundles),
            },
        },
    }
    (out / META).write_text(json.dumps(meta, indent=2, default=int))
    print(f"[prep] wrote {out}")
    return out


def read_meta(cfg: Config) -> dict[str, Any]:
    """The meta.json for the configured sample: settings it was built with, plus stats."""
    path = prepped_dir(cfg) / META
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `make prep` (or `uv run assignment-2 prep`)"
        )
    return json.loads(path.read_text())


def load_prepped(cfg: Config) -> dict[str, pd.DataFrame]:
    """{'play_train': df, 'own_valid': df, ..., 'items': df, 'bundles': df} for the sample."""
    saved = read_meta(cfg)["settings"]
    current = _settings(cfg)
    if saved != current:
        changed = sorted(k for k in current if saved.get(k) != current[k])
        raise RuntimeError(
            f"{prepped_dir(cfg)} was built with different {', '.join(changed)} settings "
            f"(cache: {[saved.get(k) for k in changed]}, config: {[current[k] for k in changed]}); "
            "run `make prep`"
        )
    out = prepped_dir(cfg)
    return {p.stem: pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))}
