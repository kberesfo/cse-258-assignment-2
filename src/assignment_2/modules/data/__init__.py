"""Data loading, sampling, splitting, and caching.

    from assignment_2.modules.config import get_config
    from assignment_2.modules.data import load_prepped

    tables = load_prepped(get_config())
    # play_{train,valid,test}, play_{valid,test}_candidates, own_{train,valid,test},
    # reviews, items, bundles

Models and evaluation should read data only through load_prepped, so every model
trains and is scored on the same cached split.
"""

from .candidates import ranking_candidates
from .prep import load_prepped, prep, read_meta
from .split import Splits
from .steam import SteamData, load_steam
from .tasks import TASKS, drop_pairs, held_out_pairs, train_tables

__all__ = [
    "TASKS",
    "Splits",
    "SteamData",
    "drop_pairs",
    "held_out_pairs",
    "load_prepped",
    "load_steam",
    "prep",
    "ranking_candidates",
    "read_meta",
    "train_tables"
]
