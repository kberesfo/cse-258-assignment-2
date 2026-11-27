"""Score the trained models and write everything the workbook and slides need to results/.

`assignment-2 eval` writes, for each task and split:
    results/metrics.json      each model's metrics, with 95% bootstrap intervals
    results/comparisons.csv   final model − each baseline, paired bootstrap intervals
    results/summaries/        prediction aggregates for plots (modules/summaries.py)
`assignment-2 figures` writes the data summaries (results/summaries/) and draws
results/figures/<task>_results.png from metrics.json.

Everything in results/ is committed, so teammates can build tables, plots and slides from
the full-data run without the full data. Per-user values stay in the artifacts cache
(eval/<task>_<split>.parquet) for further paired comparisons (load_per_user → compare).
"""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from assignment_2.modules.config import Config
from assignment_2.modules.core import compare, evaluate_engagement, evaluate_ranking
from assignment_2.modules.core.evaluate import Metric
from assignment_2.modules.data import load_prepped
from assignment_2.modules.data.prep import prepped_dir
from assignment_2.modules.data.tasks import TASKS
from assignment_2.modules.models import TASK_MODELS
from assignment_2.modules.summaries import engagement_summaries, ranking_summaries

Split = Literal["valid", "test"]
FINAL = {"play": "bpr", "engagement": "lfm"}  # compared against the task's other models
K = 10  # Hit@K, NDCG@K
N_BOOT = 1000

Summaries = dict[str, dict[str, dict[str, Metric]]]  # task -> model -> metric -> Metric


def results_path(cfg: Config, name: str) -> Path:
    """results/<name> for the full data; a dev sample gets its tag in the name
    (metrics.users_0.1.json), so 10% runs never overwrite the committed full-data results."""
    stem, ext = name.rsplit(".", 1)
    tag = "" if cfg.sample.tag == "full" else f".{cfg.sample.tag}"
    return cfg.paths.results_dir / f"{stem}{tag}.{ext}"


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
        summaries, per_user, by_model, preds = {}, {}, {}, {}
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
            summaries[name], by_model[name], preds[name] = s, pu, scores
            per_user |= {f"{name}/{m}": v for m, v in pu.items()}
        path = per_user_path(cfg, task, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_user).rename_axis("user_id").reset_index().to_parquet(path, index=False)
        if task == "play":
            ranking_summaries(cfg, split, by_model)
        else:
            engagement_summaries(cfg, split, rows, preds)
        out[task] = summaries
    return out


def write_metrics(cfg: Config, split: Split, results: Summaries) -> Path:
    """Add `split`'s results to metrics.json, keeping the other split's if from the same run
    settings (sample and seed); otherwise start fresh."""
    path = results_path(cfg, "metrics.json")
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
    data = json.loads(results_path(cfg, "metrics.json").read_text())
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


def write_comparisons(cfg: Config, split: Split, tasks: tuple[str, ...] = TASKS) -> Path:
    """Final model − each other model of the task on `split` (paired bootstrap), added to
    comparisons.csv. Rows from another sample, or for the same task and split, are replaced."""
    rows = []
    for task in tasks:
        pu, final = load_per_user(cfg, task, split), FINAL[task]
        for other in (m for m in TASK_MODELS[task] if m != final):
            for metric in pu[final]:
                d = compare(pu[final][metric], pu[other][metric], n_boot=N_BOOT, seed=cfg.seed)
                rows.append({
                    "sample": cfg.sample.tag, "task": task, "split": split,
                    "comparison": f"{final} − {other}", "metric": metric,
                    "difference": d.value, "ci_low": d.ci_low, "ci_high": d.ci_high,
                    "real": d.ci_low > 0 or d.ci_high < 0,
                })  # fmt: skip
    path = results_path(cfg, "comparisons.csv")
    new = pd.DataFrame(rows)
    if path.is_file():
        old = pd.read_csv(path)
        stale = (old["sample"] != cfg.sample.tag) | (
            (old["split"] == split) & old["task"].isin(tasks)
        )
        new = pd.concat([old[~stale], new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def plot_metrics(cfg: Config) -> list[Path]:
    """One figure per task: each metric's value with its 95% interval, per model and split."""
    from assignment_2.modules.core.plots import save, use_style

    use_style()
    m = load_metrics(cfg)
    paths = []
    for task, df in m.groupby("task", sort=False):
        metrics = list(dict.fromkeys(df["metric"]))
        models = list(dict.fromkeys(df["model"]))
        splits = list(dict.fromkeys(df["split"]))
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2))
        width = 0.8 / len(splits)
        for ax, metric in zip(axes, metrics, strict=True):
            for j, split in enumerate(splits):
                d = df[(df.metric == metric) & (df.split == split)].set_index("model")
                d = d.reindex(models)
                x = [k + (j - (len(splits) - 1) / 2) * width for k in range(len(models))]
                err = [d.value - d.ci_low, d.ci_high - d.value]
                ax.bar(x, d.value, width, yerr=err, capsize=3, label=split)
            ax.set_xticks(range(len(models)), models)
            ax.set_title(metric)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(splits))
        fig.suptitle(f"Task {'A (play)' if task == 'play' else 'B (engagement)'}: "
                     f"{cfg.sample.tag}, 95% bootstrap intervals")  # fmt: skip
        fig.tight_layout()
        paths.append(save(fig, f"{task}_results", cfg))
        plt.close(fig)
    return paths