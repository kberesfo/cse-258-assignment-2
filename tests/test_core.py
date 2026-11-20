"""Unit tests for metrics, splits, and features, using results checked by hand."""

import numpy as np
import pandas as pd
import pytest

from assignment_2.modules.core import features, metrics
from assignment_2.modules import data


def test_auc_counts_ties_as_half():
    assert metrics.calc_auc([1, 0], [0.9, 0.1]) == 1.0
    assert metrics.calc_auc([1, 0], [0.1, 0.9]) == 0.0
    assert metrics.calc_auc([1, 0], [0.5, 0.5]) == 0.5


def test_ber():
    assert metrics.calc_ber([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.25)


def test_ranking_metrics_by_hand():
    # User a's positive is ranked 2nd; user b's positive is ranked 3rd.
    users = ["a", "a", "a", "b", "b", "b"]
    labels = [0, 1, 0, 1, 0, 0]
    scores = [3, 2, 1, 1, 3, 2]
    result = metrics.ranking_per_user(users, labels, scores, k=2)
    assert result["hit@2"].tolist() == [1, 0]
    assert result["mrr"] == pytest.approx([1 / 2, 1 / 3])
    assert result["ndcg@2"] == pytest.approx([1 / np.log2(3), 0])


def test_ndcg_is_one_for_a_perfect_ranking_with_several_positives():
    result = metrics.ranking_per_user(
        ["a"] * 4, [1, 1, 0, 0], [4, 3, 2, 1], k=3)
    assert result["ndcg@3"] == pytest.approx([1.0])


def _interactions() -> pd.DataFrame:
    rows = [("heavy", f"i{n}", n) for n in range(20)]
    rows += [("light", "i0", 100), ("light", "i1", 101)]
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


def test_temporal_split_holds_out_the_latest_interactions():
    train, valid, test = data.temporal_split(_interactions(), 0.1, 0.1)
    heavy_test = test[test["user"] == "heavy"]["timestamp"]
    heavy_valid = valid[valid["user"] == "heavy"]["timestamp"]
    assert sorted(heavy_test) == [18, 19]
    assert sorted(heavy_valid) == [16, 17]
    assert len(train) + len(valid) + len(test) == 22


def test_temporal_split_keeps_sparse_users_in_train():
    train, valid, test = data.temporal_split(_interactions(), 0.1, 0.1)
    assert (train["user"] == "light").sum() == 2
    assert "light" not in set(valid["user"]) | set(test["user"])


def test_random_split_is_a_seeded_partition():
    df = _interactions()
    first = data.random_split(df, 0.2, 0.2, seed=1)
    second = data.random_split(df, 0.2, 0.2, seed=1)
    assert all(a.equals(b) for a, b in zip(first, second))
    assert sorted(pd.concat(first).index) == list(df.index)


def test_standardize_reports_missing_target_column():
    with pytest.raises(ValueError, match="rating"):
        data.standardize(pd.DataFrame({"user": [1], "item": [2]}), "rating")


def test_standardize_drops_missing_targets_and_stringifies_ids():
    df = pd.DataFrame({"user": [1, 2], "item": [3, 4], "rating": [5.0, None]})
    out = data.standardize(df, "rating")
    assert len(out) == 1
    assert out.loc[0, "user"] == "1"


def test_ranking_candidates_never_sample_seen_items():
    seen = pd.DataFrame({"user": ["a", "a", "b"], "item": ["i0", "i1", "i2"]})
    held = seen[seen["item"] == "i1"]
    table = features.ranking_candidates(held, seen, n_negatives=5, seed=0)
    assert set(table[table["label"] == 1]["item"]) == {"i1"}
    assert set(table[table["label"] == 0]["item"]) == {"i2"}


def test_interaction_matrix_ignores_unknown_ids_and_repeats():
    train = pd.DataFrame({"user": ["a", "a"], "item": ["x", "x"]})
    index = features.IdIndex.from_frame(train)
    extra = pd.DataFrame({"user": ["a", "zzz"], "item": ["x", "x"]})
    matrix = features.interaction_matrix(pd.concat([train, extra]), index)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 1
