"""Shared figure style and saving, so every notebook's plots look like one project's.

    from assignment_2.modules.core.plots import use_style, save

    use_style()                       # once, at the top of a notebook
    fig, ax = plt.subplots()
    ...
    save(fig, "games_per_user")       # -> results/figures/games_per_user.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from assignment_2.modules.config import Config, get_config

# Okabe-Ito palette: distinguishable with common color-vision deficiencies and in grayscale.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def use_style() -> None:
    """Set matplotlib defaults for every figure in the project."""
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "figure.dpi": 110,  # on-screen sharpness; save() sets its own dpi
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=PALETTE),
            "legend.frameon": False,
        }
    )


def save(fig: Figure, name: str, cfg: Config | None = None) -> Path:
    """Save `fig` as results/figures/<name>.png and return the path.

    Dev-sample figures get the sample tag in the name (games_per_user.users_0.1.png),
    so they never overwrite, or get mistaken for, the final full-data figure.
    """
    if cfg is None:
        cfg = get_config()

    filename = name
    if cfg.sample.tag != "full":
        filename = f"{filename}.{cfg.sample.tag}"
    filename = f"{filename}.png"

    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.figures / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def log_hist(
    ax: Axes,
    values: ArrayLike,
    xlabel: str,
    ylabel: str,
    title: str = "",
    n_bins: int = 40,
    log_y: bool = False,
) -> Axes:
    """Histogram with log-spaced bins on a log x-axis, for long-tailed counts. Returns ax."""
    values = np.asarray(values, dtype=float)
    if (values <= 0).any():
        raise ValueError("log_hist needs values > 0; filter out zeros first")
    bins = np.geomspace(values.min(), values.max(), n_bins + 1)
    ax.hist(values, bins=bins)
    ax.set(xscale="log", yscale="log" if log_y else "linear",
           xlabel=xlabel, ylabel=ylabel, title=title)  # fmt: skip

    return ax
