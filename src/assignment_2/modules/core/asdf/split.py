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

from assignment_2.modules.config.paths import Split


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        n = len(self.train) + len(self.valid) + len(self.test)
        parts = [
            f"{name}: {len(df):,} ({len(df) / n:.1%}), {df.user_id.nunique():,} users"
            for name, df in (("train", self.train), ("valid", self.valid), ("test", self.test))
        ]
        return " | ".join(parts)


def split_by_user(
    df: pd.DataFrame,
    split: Split,
    seed: int,
    time_col: str | None = None,
    min_per_user: int = 3,
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
    n_user = df.groupby("user_id", sort=False)["user_id"].transform("size").to_numpy()
    # Position from the end of each user's block: 0 = last (latest / random pick).
    from_end = df.groupby("user_id", sort=False).cumcount(ascending=False).to_numpy()
    # Undated rows are never held out, so count holdouts from the dated rows only.
    n_dated = pd.Series(dated).groupby(user).transform("sum").to_numpy()

    n_test = np.where(n_user >= min_per_user, np.floor(n_dated * split.test_frac), 0)
    n_valid = np.where(n_user >= min_per_user, np.floor(n_dated * split.valid_frac), 0)
    # Every eligible user gets at least one test and one valid row when fractions > 0.
    eligible = (n_user >= min_per_user) & (n_dated >= 2)
    n_test = np.where(eligible & (split.test_frac > 0), np.maximum(n_test, 1), 0)
    n_valid = np.where(eligible & (split.valid_frac > 0), np.maximum(n_valid, 1), 0)

    is_test = from_end < n_test
    is_valid = ~is_test & (from_end < n_test + n_valid)
    is_train = ~(is_test | is_valid)

    keep = df.columns
    return Splits(
        train=df[is_train][keep].reset_index(drop=True),
        valid=df[is_valid][keep].reset_index(drop=True),
        test=df[is_test][keep].reset_index(drop=True),
    )
