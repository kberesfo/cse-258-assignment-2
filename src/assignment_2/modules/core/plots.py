"""Figures shared by the notebook and the slides."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from assignment_2.modules.config import Config
from assignment_2.modules.core.data import SPLITS, load_split

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def _figure(ncols: int = 1, width: float = 6.5, height: float = 3.6) -> tuple[Figure, list]:
    fig = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE, dpi=150)
    axes = [fig.add_subplot(1, ncols, i + 1) for i in range(ncols)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=9)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.yaxis.label.set_color(TEXT_MUTED)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def _title(ax, text: str) -> None:
    ax.set_title(text, color=TEXT, fontsize=11, loc="left")


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, facecolor=SURFACE)


def _target_distribution(df: pd.DataFrame, column: str, path: Path) -> None:
    counts = df[column].value_counts().sort_index()
    fig, (ax,) = _figure()
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=SERIES, width=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(column)
    ax.set_ylabel("Interactions")
    _title(ax, f"Distribution of {column}")
    _save(fig, path)


def _long_tail(df: pd.DataFrame, path: Path) -> None:
    fig, axes = _figure(ncols=2)
    for ax, column, name in zip(axes, ("user", "item"), ("users", "items")):
        counts = np.sort(df[column].value_counts().to_numpy())[::-1]
        ax.plot(np.arange(1, len(counts) + 1), counts, color=SERIES, linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"Rank among {name}")
        ax.set_ylabel("Interactions")
        _title(ax, f"Activity across {name}")
    _save(fig, path)


def _timeline(df: pd.DataFrame, path: Path) -> None:
    months = pd.to_datetime(df["timestamp"], unit="s").dt.to_period("M")
    counts = months.value_counts().sort_index()
    fig, (ax,) = _figure()
    ax.plot(counts.index.to_timestamp(), counts.to_numpy(), color=SERIES, linewidth=2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Interactions per month")
    _title(ax, "Interactions over time")
    _save(fig, path)


def _model_comparison(split: str, entry: dict, path: Path) -> None:
    models = sorted(entry["models"])
    metrics = list(entry["models"][models[0]])
    height = max(2.4, 0.5 * len(models) + 1.4)
    fig, axes = _figure(ncols=len(metrics), width=3.2 * len(metrics) + 0.6, height=height)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        results = [entry["models"][m][metric] for m in models]
        values = np.array([r["value"] for r in results])
        y = np.arange(len(models))
        if results[0]["ci_low"] is not None:
            low = values - np.array([r["ci_low"] for r in results])
            high = np.array([r["ci_high"] for r in results]) - values
            ax.errorbar(values, y, xerr=[low, high], fmt="none", ecolor=SERIES, linewidth=2)
        ax.scatter(values, y, s=64, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_yticks(y, models if i == 0 else [])
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.invert_yaxis()
        ax.margins(x=0.15)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(metric)
    interval = " with 95% bootstrap intervals" if results[0]["ci_low"] is not None else ""
    fig.suptitle(
        f"Model comparison on {split}{interval}", color=TEXT, fontsize=11, x=0.01, ha="left"
    )
    _save(fig, path)


def figures(cfg: Config) -> None:
    """Write PNGs to cfg.paths.figures: eda_* from the data, models_* from the metrics."""
    out = cfg.paths.figures
    df = pd.concat([load_split(cfg, name) for name in SPLITS])
    for column in ("rating", "label"):
        if column in df.columns:
            _target_distribution(df, column, out / f"eda_{column}.png")
    _long_tail(df, out / "eda_long_tail.png")
    if "timestamp" in df.columns:
        _timeline(df, out / "eda_timeline.png")

    if cfg.paths.metrics.is_file():
        for split, entry in json.loads(cfg.paths.metrics.read_text()).items():
            _model_comparison(split, entry, out / f"models_{split}.png")
