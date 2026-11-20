"""Loader for Steam v1 (Australian users) + bundles + game metadata.

Raw files (from scripts/get_data steam) are nested: one record per *user* (or bundle),
holding a list of items. This flattens them into tidy tables, one row per event.
IDs are strings in every table so joins never silently fail on int vs str.

    data = load_steam(cfg.paths.data_dir / "steam", user_frac=0.1, seed=cfg.seed)

user_frac < 1 keeps a deterministic sample of users (whole histories, same users in
reviews and interactions); unsampled lines are skipped before parsing. Items and
bundles are always loaded in full (they're small).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .records import records, user_sampler

REVIEWS = "australian_user_reviews.json.gz"
USERS_ITEMS = "australian_users_items.json.gz"
BUNDLES = "bundle_data.json.gz"
GAMES = "steam_games.json.gz"


@dataclass(frozen=True)
class SteamData:
    interactions: pd.DataFrame
    reviews: pd.DataFrame
    items: pd.DataFrame
    bundles: pd.DataFrame

    def summary(self) -> str:
        i, r = self.interactions, self.reviews
        return (
            f"interactions: {len(i):,} rows, {i.user_id.nunique():,} users, "
            f"{i.item_id.nunique():,} items | reviews: {len(r):,} "
            f"({r.posted.isna().mean():.1%} with no parsable date) | "
            f"items: {len(self.items):,} | bundles: {self.bundles.bundle_id.nunique():,}"
        )


# ── field parsers ───────────────────────────────────────────────────────────

_HELPFUL = re.compile(r"([\d,]+) of ([\d,]+) people")
_POSTED = re.compile(r"Posted\s+(\w+ \d{1,2})(?:,\s*(\d{4}))?")
_RELEASE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y", "%B %d, %Y", "%b %Y", "%B %Y", "%Y")


def parse_release_date(v: object) -> pd.Timestamp:
    """Steam release dates come in several formats; unknown ones ('Soon..', 'TBA') -> NaT."""
    if not isinstance(v, str):
        return pd.NaT
    s = v.strip()
    for fmt in _RELEASE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return pd.NaT


def parse_posted(s: str | None) -> pd.Timestamp:
    """'Posted November 5, 2011.' -> 2011-11-05. Dates without a year -> NaT.

    Reviews from the year of the crawl show no year on Steam, so the year is unknown.
    They are kept (NaT), not guessed; decide how to handle them in EDA.
    """
    m = _POSTED.search(s or "")
    if not m or not m.group(2):
        return pd.NaT
    try:
        return pd.Timestamp(datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %d %Y"))
    except ValueError:
        return pd.NaT


def parse_helpful(s: str | None) -> tuple[int, int]:
    """'3 of 4 people (75%) found this review helpful' -> (3, 4); 'No ratings yet' -> (0, 0)."""
    m = _HELPFUL.search(s or "")
    if not m:
        return 0, 0
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def parse_funny(s: str | None) -> int:
    """'2 people found this review funny' -> 2; '' -> 0."""
    m = re.search(r"([\d,]+)", s or "")
    return int(m.group(1).replace(",", "")) if m else 0


def parse_price(v: object) -> float:
    """4.99 -> 4.99; 'Free To Play' / 'Free' -> 0.0; other strings or missing -> NaN."""
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str) and "free" in v.lower():
        return 0.0
    return np.nan


def parse_money(v: object) -> float:
    """'$29.66' -> 29.66; numbers pass through; 'Free...' -> 0.0; anything else -> NaN."""
    if isinstance(v, str):
        try:
            return float(v.replace("$", "").replace(",", "").strip())
        except ValueError:
            pass
    return parse_price(v)


def parse_percent(v: object) -> float:
    """'10%' -> 0.10; missing or unparsable -> NaN."""
    m = re.search(r"([\d.]+)\s*%", v) if isinstance(v, str) else None
    return float(m.group(1)) / 100 if m else np.nan


def split_genres(v: object) -> list[str]:
    """'Casual, Indie' -> ['Casual', 'Indie'] (same list form as items.genres)."""
    if isinstance(v, list):
        return [str(g).strip() for g in v]
    return [g.strip() for g in v.split(",") if g.strip()] if isinstance(v, str) else []


# ── tables ──────────────────────────────────────────────────────────────────


def load_interactions(path: Path, keep_user: Callable[[str], bool] | None = None) -> pd.DataFrame:
    rows = [
        (u["user_id"], str(it["item_id"]), it.get("playtime_forever", 0),
         it.get("playtime_2weeks", 0))
        for u in records(path, keep_user)
        for it in u.get("items", [])
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "playtime_forever", "playtime_2weeks"])
    return df.astype({"playtime_forever": "int32", "playtime_2weeks": "int32"})


def load_reviews(path: Path, keep_user: Callable[[str], bool] | None = None) -> pd.DataFrame:
    rows = []
    for u in records(path, keep_user):
        for r in u.get("reviews", []):
            yes, total = parse_helpful(r.get("helpful"))
            rows.append(
                (
                    u["user_id"],
                    str(r["item_id"]),
                    bool(r.get("recommend")),
                    parse_posted(r.get("posted")),
                    r.get("review", ""),
                    parse_funny(r.get("funny")),
                    yes,
                    total,
                )
            )
    cols = ["user_id", "item_id", "recommend", "posted", "text", "funny",
            "helpful_yes", "helpful_total"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def load_items(path: Path) -> pd.DataFrame:
    rows = [
        {
            "item_id": str(g["id"]),
            "name": g.get("app_name") or g.get("title"),
            "developer": g.get("developer"),
            "publisher": g.get("publisher"),
            "genres": list(g.get("genres") or []),
            "tags": list(g.get("tags") or []),
            "specs": list(g.get("specs") or []),
            "price": parse_price(g.get("price")),
            "release_date": parse_release_date(g.get("release_date")),
            "early_access": bool(g.get("early_access", False)),
        }
        for g in records(path)
        # a few metadata rows have no id and can't be joined
        if g.get("id") is not None
    ]
    return pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)


def load_bundles(path: Path) -> pd.DataFrame:
    """One row per (bundle, game). Bundle-level columns repeat on each of its games."""
    rows = []
    for b in records(path):
        bundle = (
            str(b["bundle_id"]),
            b.get("bundle_name"),
            parse_money(b.get("bundle_price")),
            parse_money(b.get("bundle_final_price")),
            parse_percent(b.get("bundle_discount")),
            len(b.get("items", [])),
        )
        for it in b.get("items", []):
            rows.append(
                (
                    *bundle,
                    str(it["item_id"]),
                    it.get("item_name"),
                    split_genres(it.get("genre")),
                    parse_money(it.get("discounted_price")),
                )
            )

    cols = [
        "bundle_id",
        "bundle_name",
        "bundle_price",
        "bundle_final_price",
        "bundle_discount",
        "bundle_size",
        "item_id",
        "item_name",
        "item_genres",
        "item_price",
    ]

    return pd.DataFrame(rows, columns=cols)


def load_steam(data_dir: Path, user_frac: float = 1.0, seed: int = 0) -> SteamData:
    data_dir = Path(data_dir)
    missing = [f for f in (REVIEWS, USERS_ITEMS, BUNDLES, GAMES) if not (data_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir}; run `uv run python scripts/get_data steam`"
        )
    keep = user_sampler(user_frac, seed)
    return SteamData(
        interactions=load_interactions(data_dir / USERS_ITEMS, keep),
        reviews=load_reviews(data_dir / REVIEWS, keep),
        items=load_items(data_dir / GAMES),
        bundles=load_bundles(data_dir / BUNDLES),
    )
