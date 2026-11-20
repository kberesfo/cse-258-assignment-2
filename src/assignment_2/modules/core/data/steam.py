from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SteamData:
    interactions: pd.DataFrame
    reviews: pd.DataFrame
    items: pd.DataFrame
    bundles: pd.DataFrame

    def summary(self) -> str:
        i, r = self.interactions, self.reviews
        return (
            f"interactions: {len(i):,} rows, {i.user_id.nunique():,} users, "
            f"{i.item_id.nunique():,} items | reviews: {len(r):,} "
            f"({r.posted.isna().mean():.1%} with no parsable date) | "
            f"items: {len(self.items):,} | bundles: {self.bundles.bundle_id.nunique():,}"
        )
