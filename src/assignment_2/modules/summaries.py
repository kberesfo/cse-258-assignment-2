"""Small CSV summaries of the data and of the final models' predictions, for plots and
tables in the workbook and slides (results/summaries/).

Everything here is an aggregate (counts, per-game rows, bucket tables), small enough to
commit, so notebooks can plot the full-data results without loading the full data.
Written by `assignment-2 figures` (data) and `assignment-2 eval` (predictions). Files from a
dev sample get its tag in the name (per_game.users_0.1.csv), as figures do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core import calc_auc
from assignment_2.modules.data import load_prepped, read_meta

# log(1 + minutes) buckets for task B's error analysis
BUCKETS = [-np.inf, 0, 60, 600, 6000, np.inf]
BUCKET_LABELS = ["never (0)", "≤ 1 h", "1–10 h", "10–100 h", "100 h+"]


def summary_path(cfg: Config, name: str) -> Path:
    """results/summaries/<name>.csv; dev samples get the tag in the name."""
    tag = "" if cfg.sample.tag == "full" else f".{cfg.sample.tag}"
    folder = cfg.paths.results_dir / "summaries"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}{tag}.csv"


def _write(cfg: Config, name: str, df: pd.DataFrame) -> Path:
    path = summary_path(cfg, name)
    df.to_csv(path, index=False)
    return path


def read_summary(cfg: Config, name: str) -> pd.DataFrame:
    """Read results/summaries/<name>.csv for the configured sample."""
    return pd.read_csv(summary_path(cfg, name))


# ── data (EDA) ──────────────────────────────────────────────────────────────


def data_summaries(cfg: Config) -> list[Path]:
    t = load_prepped(cfg)
    owned, reviews = t["owned"], t["reviews"]
    played = owned.playtime_forever > 0
    log_min = np.log1p(owned.playtime_forever)

    per_game = (
        owned.assign(played=played, played_120=owned.playtime_forever >= 120, log_min=log_min)
        .groupby("item_id")
        .agg(
            owners=("user_id", "size"),
            players=("played", "sum"),
            players_120=("played_120", "sum"),
            median_log_minutes=("log_min", "median"),
            median_hours=("playtime_forever", lambda s: s.median() / 60),
        )
        .assign(play_rate=lambda d: d.players / d.owners)
    )
    played_log = log_min[played].groupby(owned.item_id[played]).median()
    per_game["played_median_log_minutes"] = played_log.reindex(per_game.index)
    per_game["in_bundle"] = per_game.index.isin(t["bundles"].item_id)
    per_game["has_metadata"] = per_game.index.isin(t["items"].item_id)
    names = t["items"].set_index("item_id")["name"]
    per_game.insert(0, "name", names.reindex(per_game.index))

    per_user = (  # no user ids: library size and play habits only
        owned.assign(played=played, played_120=owned.playtime_forever >= 120)
        .groupby("user_id")
        .agg(
            games=("item_id", "size"),
            played=("played", "sum"),
            played_120=("played_120", "sum"),
            total_hours=("playtime_forever", lambda s: s.sum() / 60),
        )
        .assign(share_never_played=lambda d: 1 - d.played / d.games)
        .reset_index(drop=True)
    )

    hours = owned.playtime_forever[played] / 60
    edges = np.geomspace(hours.min(), hours.max(), 41)
    counts, _ = np.histogram(hours, bins=edges)
    playtime_hist = pd.DataFrame({"hours_lo": edges[:-1], "hours_hi": edges[1:], "games": counts})

    years = reviews.posted.dt.year.value_counts(dropna=False).sort_index()
    reviews_per_year = pd.DataFrame(
        {"year": [str(int(y)) if pd.notna(y) else "no year" for y in years.index],
         "reviews": years.to_numpy()}
    )  # fmt: skip

    stats = read_meta(cfg)["stats"]
    dataset = pd.json_normalize(stats, sep=".").T.reset_index()
    dataset.columns = ["stat", "value"]

    return [
        _write(cfg, "per_game", per_game.reset_index()),
        _write(cfg, "per_user", per_user),
        _write(cfg, "playtime_played_hist", playtime_hist),
        _write(cfg, "reviews_per_year", reviews_per_year),
        _write(cfg, "dataset_stats", dataset),
    ]


# ── predictions (results) ───────────────────────────────────────────────────


def ranking_summaries(cfg: Config, split: str, per_user: dict[str, dict]) -> list[Path]:
    """Task A: how many users got their held-out game at each rank, per model (rank = 1/MRR
    with one positive per user). Enough for Hit@k at any k, median rank, top-1 rate."""
    rows = []
    for model, pu in per_user.items():
        ranks = (1 / pu["mrr"]).round().astype(int)
        counts = ranks.value_counts().sort_index()
        rows += [{"model": model, "rank": r, "users": n} for r, n in counts.items()]
    return [_write(cfg, f"play_{split}_ranks", pd.DataFrame(rows))]


def engagement_summaries(
    cfg: Config, split: str, rows: pd.DataFrame, preds: dict[str, np.ndarray]
) -> list[Path]:
    """Task B error analysis (row-level): error by playtime bucket, calibration by
    prediction decile, histograms of truth vs predictions, and headline numbers."""
    y = np.log1p(rows["playtime_forever"].to_numpy(float))
    zero = y == 0
    bucket = pd.cut(np.expm1(y), BUCKETS, labels=BUCKET_LABELS)
    edges = np.linspace(0, max(y.max(), *(p.max() for p in preds.values())), 41)
    hist = {"log_minutes_lo": edges[:-1], "log_minutes_hi": edges[1:],
            "true": np.histogram(y, bins=edges)[0]}  # fmt: skip

    buckets, calibration, headline = [], [], []
    for model, p in preds.items():
        err = p - y
        df = pd.DataFrame({"bucket": bucket, "err": err, "err2": err**2, "p": p, "y": y})
        g = df.groupby("bucket", observed=True)
        buckets.append(
            pd.DataFrame({
                "model": model,
                "share_of_rows": g.size() / len(df),
                "median_pred_minutes": np.expm1(g.p.median()),
                "mean_error_log": g.err.mean(),
                "mse": g.err2.mean(),
                "share_of_squared_error": g.err2.sum() / df.err2.sum(),
            }).reset_index()
        )  # fmt: skip
        decile = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = df.groupby(decile).agg(mean_pred=("p", "mean"), mean_true=("y", "mean"),
                                     rows=("y", "size"))  # fmt: skip
        calibration.append(cal.rename_axis("decile").reset_index().assign(model=model))
        hist[model] = np.histogram(p, bins=edges)[0]
        ratio = np.exp(np.abs(err))
        headline.append({
            "model": model,
            "mse": np.mean(err**2),
            "mse_played_only": np.mean(err[~zero] ** 2),
            "mse_if_zeros_perfect": np.mean((np.where(zero, 0, p) - y) ** 2),
            "median_ratio": np.median(ratio),
            "within_2x": np.mean(ratio <= 2),
            "within_10x": np.mean(ratio <= 10),
            # how well the prediction ranks never-played games below played ones
            "auc_never_played": calc_auc(zero, -p) if 0 < zero.sum() < len(zero) else np.nan,
            "pred_std": p.std(),
            "true_std": y.std(),
        })  # fmt: skip

    return [
        _write(cfg, f"engagement_{split}_buckets", pd.concat(buckets, ignore_index=True)),
        _write(cfg, f"engagement_{split}_calibration", pd.concat(calibration, ignore_index=True)),
        _write(cfg, f"engagement_{split}_hist", pd.DataFrame(hist)),
        _write(cfg, f"engagement_{split}_headline", pd.DataFrame(headline)),
    ]