# core/data/__init__.py
"""Data loading, sampling, splitting, and caching.

    from assignment_2.modules.core.data import load_prepped
    tables = load_prepped(get_config())          # cached train/valid/test + items/bundles
"""

from .prep import load_prepped, prep
from .split import Splits
from .steam import SteamData, load_steam

__all__ = ["Splits", "SteamData", "load_prepped", "load_steam", "prep"]
