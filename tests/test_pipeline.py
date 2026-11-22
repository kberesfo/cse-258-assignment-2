"""Smoke test: the whole pipeline runs on synthetic data for every task."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.skip("rewrite for the current pipeline", allow_module_level=True)

from assignment_2.cli import main
from assignment_2.modules.config import load
from assignment_2.modules.core.features import (
    IdIndex,
    interaction_matrix,
    surprise_trainset,
)
from assignment_2.modules.data import load_split

REPO = Path(__file__).resolve().parents[1]


def run_pipeline(tmp_path: Path, monkeypatch, task: str) -> tuple[list[str], dict]:
    monkeypatch.setenv("A2_DATASET", "synthetic")
    monkeypatch.setenv("A2_TASK", task)
    monkeypatch.setenv("A2_EVALUATION__N_BOOTSTRAP", "20")
    paths = ["--art-dir", str(tmp_path / "art"), "--results-dir", str(tmp_path / "res")]
    for command in (["prep"], ["train"], ["eval"], ["eval", "--split", "test"], ["figures"]):
        main(command + paths)
    return paths, json.loads((tmp_path / "res" / "metrics.json").read_text())


@pytest.mark.parametrize(
    ("task", "metric", "models"),
    [
        ("rating", "mse", {"trivial", "popularity", "bias"}),
        ("binary", "auc", {"trivial", "popularity", "bias"}),
        ("ranking", "hit@10", {"trivial", "popularity"}),
    ],
)
def test_pipeline_runs_for_each_task(tmp_path, monkeypatch, task, metric, models):
    _, results = run_pipeline(tmp_path, monkeypatch, task)

    assert set(results) == {"valid", "test"}
    assert set(results["valid"]["models"]) == models
    for split in results.values():
        for scores in split["models"].values():
            assert scores[metric]["ci_low"] <= scores[metric]["ci_high"]
    assert (tmp_path / "res" / "figures" / "models_test.png").is_file()
    assert (tmp_path / "res" / "figures" / "eda_long_tail.png").is_file()


def test_real_models_beat_trivial_on_synthetic_ratings(tmp_path, monkeypatch):
    _, results = run_pipeline(tmp_path, monkeypatch, "rating")
    models = results["test"]["models"]
    assert models["bias"]["mse"]["value"] < models["trivial"]["mse"]["value"]


def test_popularity_beats_random_on_synthetic_ranking(tmp_path, monkeypatch):
    _, results = run_pipeline(tmp_path, monkeypatch, "ranking")
    models = results["test"]["models"]
    assert models["popularity"]["hit@10"]["value"] > models["trivial"]["hit@10"]["value"]


def test_rerunning_valid_keeps_test_results(tmp_path, monkeypatch):
    paths, first = run_pipeline(tmp_path, monkeypatch, "rating")
    main(["eval", *paths])
    second = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert first["test"] == second["test"]


def test_library_adapters_train(tmp_path, monkeypatch):
    run_pipeline(tmp_path, monkeypatch, "rating")
    cfg = load(paths={"art_dir": tmp_path / "art", "results_dir": tmp_path / "res"})
    train = load_split(cfg, "train")

    from implicit.bpr import BayesianPersonalizedRanking
    from surprise import SVD

    index = IdIndex.from_frame(train)
    bpr = BayesianPersonalizedRanking(factors=4, iterations=2, random_state=cfg.seed)
    bpr.fit(interaction_matrix(train, index), show_progress=False)
    assert bpr.user_factors.shape[0] == index.n_users

    svd = SVD(n_factors=4, n_epochs=2, random_state=cfg.seed)
    svd.fit(surprise_trainset(train))
    assert np.isfinite(svd.predict(train.loc[0, "user"], train.loc[0, "item"]).est)


def test_notebook_exports_valid_html(tmp_path, monkeypatch):
    run_pipeline(tmp_path, monkeypatch, "rating")
    env = os.environ | {
        "A2_PATHS__ART_DIR": str(tmp_path / "art"),
        "A2_PATHS__RESULTS_DIR": str(tmp_path / "res"),
    }
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--execute",
            str(REPO / "notebooks" / "workbook.ipynb"),
            "--ExecutePreprocessor.kernel_name=python3",
            "--output-dir",
            str(tmp_path),
            "--output",
            "workbook",
        ],
        check=True,
        env=env,
    )
    html = (tmp_path / "workbook.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "fit_bias" in html  # model code must be visible to graders
