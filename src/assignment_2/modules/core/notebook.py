"""Display helpers for workbook.ipynb. Graders only see the exported HTML."""

import inspect
import json

import pandas as pd
from IPython.display import Image, Markdown, display

from assignment_2.modules.config import Config


def show(obj: object) -> None:
    """Print an object's source code, so the implementation appears in the exported HTML."""
    display(Markdown(f"```python\n{inspect.getsource(obj).rstrip()}\n```"))


def show_figures(cfg: Config, prefix: str) -> None:
    """Display every saved figure whose name starts with `prefix`."""
    for path in sorted(cfg.paths.figures.glob(f"{prefix}*.png")):
        display(Image(filename=path, width=640))


def metrics_table(cfg: Config, split: str) -> pd.DataFrame:
    """One row per model and one column per metric, as `value [low, high]`."""

    def cell(result: dict) -> str:
        if result["ci_low"] is None:
            return f"{result['value']:.4f}"
        return f"{result['value']:.4f} [{result['ci_low']:.4f}, {result['ci_high']:.4f}]"

    models = json.loads(cfg.paths.metrics.read_text())[split]["models"]
    rows = {name: {m: cell(r) for m, r in metrics.items()} for name, metrics in models.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def show_metrics(cfg: Config) -> None:
    """Display the metrics table for each evaluated split."""
    if not cfg.paths.metrics.is_file():
        display(Markdown("_No metrics yet: run `make results`._"))
        return
    for split in json.loads(cfg.paths.metrics.read_text()):
        display(Markdown(f"**{split}**"))
        display(metrics_table(cfg, split))
