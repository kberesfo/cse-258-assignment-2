"""Data loading, sampling, splitting, and caching.

    from assignment_2.modules.config import get_config
    from assignment_2.modules.core.data import load_prepped

    tables = load_prepped(get_config())   # interactions_{train,valid,test}, reviews_*, items, bundles

Models and evaluation should read data only through load_prepped, so every model
trains and is scored on the same cached split.
"""

from .prep import load_prepped, prep
from .split import Splits
from .steam import SteamData, load_steam

__all__ = ["Splits", "SteamData", "load_prepped", "load_steam", "prep"]
