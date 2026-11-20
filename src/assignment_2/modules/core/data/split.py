"""Train / valid / test splits, done within each user.

Every user with enough history appears in all three splits, with some of their
interactions held out. That's the setup for "given what we know about this user,
predict their other items" tasks. Users with fewer than `min_per_user` rows stay
entirely in train (too little history to hold anything out).

If `by_time` is set and a time column is given, each user's *latest* rows are held
out; rows with no timestamp stay in train. Otherwise rows are held out at random.
The Steam ownership data has no timestamps, so it always splits at random.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from assignment_2.modules.config.split import Split


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    n_cold_dropped: int = 0  # held-out rows removed because their item is never in train

    def summary(self) -> str:
        n = len(self.train) + len(self.valid) + len(self.test)
        parts = [
            f"{name}: {len(df):,} ({len(df) / n:.1%}), {df.user_id.nunique():,} users"
            for name, df in (("train", self.train), ("valid", self.valid), ("test", self.test))
        ]
        if self.n_cold_dropped:
            parts.append(f"cold items dropped: {self.n_cold_dropped:,}")
        return " | ".join(parts)


def split_by_user(
    df: pd.DataFrame,
    split: Split,
    seed: int,
    time_col: str | None = None,
) -> Splits:
    rng = np.random.default_rng(seed)
    use_time = split.by_time and time_col is not None and time_col in df.columns

    if use_time:
        # Latest last; NaT sorts first so undated rows are never held out.
        order = df[time_col].fillna(pd.Timestamp.min).to_numpy()
        tiebreak = rng.random(len(df))
        df = df.iloc[np.lexsort((tiebreak, order, df["user_id"].to_numpy()))]
        dated = df[time_col].notna().to_numpy()
    else:
        df = df.iloc[rng.permutation(len(df))]
        df = df.iloc[np.argsort(df["user_id"].to_numpy(), kind="stable")]
        dated = np.ones(len(df), dtype=bool)

    user = df["user_id"].to_numpy()
    by_user = df.groupby("user_id", sort=False)
    n_user = by_user["user_id"].transform("size").to_numpy()
    # Position from the end of each user's block: 0 = last (latest / random pick).
    from_end = by_user.cumcount(ascending=False).to_numpy()
    # Undated rows are never held out, so count holdouts from the dated rows only.
    n_dated = pd.Series(dated).groupby(user).transform("sum").to_numpy()

    # Eligible users give up >= 1 valid and >= 1 test row and keep >= 1 train row.
    # min_per_user >= 3 (enforced in config) guarantees the train row.
    eligible = (n_user >= split.min_per_user) & (n_dated >= 2)
    n_test = np.where(
        eligible & (split.test_frac > 0), np.maximum(
            np.floor(n_dated * split.test_frac), 1), 0
    )
    n_valid = np.where(
        eligible & (split.valid_frac > 0),
        # fmt: skip
        np.maximum(np.floor(n_dated * split.valid_frac), 1), 0)

    is_test = from_end < n_test
    is_valid = ~is_test & (from_end < n_test + n_valid)
    train = df[~(is_test | is_valid)].reset_index(drop=True)
    valid = df[is_valid].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)

    n_cold = 0
    if split.drop_cold:
        # A model can't score an item it never saw in training; every held-out user
        # already has train rows by construction, so only items need checking.
        seen = train["item_id"].unique()
        keep_v, keep_t = valid["item_id"].isin(
            seen), test["item_id"].isin(seen)
        n_cold = int((~keep_v).sum() + (~keep_t).sum())
        valid = valid[keep_v].reset_index(drop=True)
        test = test[keep_t].reset_index(drop=True)

    return Splits(train=train, valid=valid, test=test, n_cold_dropped=n_cold)
