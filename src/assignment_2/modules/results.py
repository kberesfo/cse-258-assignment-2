"""Score the trained models on a fixed split and save the results (`assignment-2 eval`).

results/metrics.json holds every task × split × model summary (value and 95% bootstrap
interval). Per-user values go to the artifacts cache (eval/<task>_<split>.parquet), so
notebooks can run paired comparisons (core.compare) without refitting anything.
"""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core import evaluate_engagement, evaluate_ranking
from assignment_2.modules.core.evaluate import Metric
from assignment_2.modules.data import load_prepped
from assignment_2.modules.data.prep import prepped_dir
from assignment_2.modules.data.tasks import TASKS
from assignment_2.modules.models import TASK_MODELS

Split = Literal["valid", "test"]
K = 10  # Hit@K, NDCG@K
N_BOOT = 1000

Summaries = dict[str, dict[str, dict[str, Metric]]]  # task -> model -> metric -> Metric


def eval_rows(t: dict[str, pd.DataFrame], task: str, split: Split) -> pd.DataFrame:
    """The fixed rows a task is scored on: task A's candidate table, task B's held-out games."""
    return t[f"play_{split}_candidates"] if task == "play" else t[f"own_{split}"]


def model_path(cfg: Config, task: str, name: str) -> Path:
    return prepped_dir(cfg) / "models" / task / f"{name}.pkl"


def per_user_path(cfg: Config, task: str, split: Split) -> Path:
    return prepped_dir(cfg) / "eval" / f"{task}_{split}.parquet"


def score_trained(cfg: Config, split: Split, tasks: tuple[str, ...] = TASKS) -> Summaries:
    """Score each task's TASK_MODELS (trained by `assignment-2 train`) on `split`."""
    t = load_prepped(cfg)
    out: Summaries = {}
    for task in tasks:
        rows = eval_rows(t, task, split)
        summaries, per_user = {}, {}
        for name in TASK_MODELS[task]:
            path = model_path(cfg, task, name)
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing: run `assignment-2 train --task {task}`")
            with path.open("rb") as f:
                model = pickle.load(f)
            scores = model.predict(rows)
            if task == "play":
                s, pu = evaluate_ranking(rows, scores, k=K, n_boot=N_BOOT, seed=cfg.seed)
            else:
                s, pu = evaluate_engagement(rows, scores, n_boot=N_BOOT, seed=cfg.seed)
            summaries[name] = s
            per_user |= {f"{name}/{m}": v for m, v in pu.items()}
        path = per_user_path(cfg, task, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_user).rename_axis("user_id").reset_index().to_parquet(path, index=False)
        out[task] = summaries
    return out


def write_metrics(cfg: Config, split: Split, results: Summaries) -> Path:
    """Add `split`'s results to metrics.json, keeping the other split's if from the same run
    settings (sample and seed); otherwise start fresh."""
    path = cfg.paths.metrics
    run = {"sample": cfg.sample.tag, "seed": cfg.seed}
    data = json.loads(path.read_text()) if path.is_file() else {}
    if {k: data.get(k) for k in run} != run:
        data = dict(run)
    for task, models in results.items():
        data.setdefault(task, {})[split] = {
            name: {m: asdict(v) for m, v in s.items()} for name, s in models.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_metrics(cfg: Config) -> pd.DataFrame:
    """metrics.json as a table: one row per (task, split, model), value and 95% CI columns."""
    data = json.loads(cfg.paths.metrics.read_text())
    rows = [
        {"task": task, "split": split, "model": name, "metric": m, **v}
        for task in TASKS
        for split, models in data.get(task, {}).items()
        for name, metrics in models.items()
        for m, v in metrics.items()
    ]
    return pd.DataFrame(rows)


def load_per_user(cfg: Config, task: str, split: Split) -> dict[str, dict[str, pd.Series]]:
    """{model: {metric: per-user Series}}, the shape core.compare expects."""
    df = pd.read_parquet(per_user_path(cfg, task, split)).set_index("user_id")
    out: dict[str, dict[str, pd.Series]] = {}
    for col in df.columns:
        name, metric = col.split("/", 1)
        out.setdefault(name, {})[metric] = df[col].rename(metric)
    return out