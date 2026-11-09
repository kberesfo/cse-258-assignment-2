from assignment_2.modules.config import Config
from assignment_2.modules.models import baselines
from assignment_2.modules.models.base import SkipModel

MODELS = {
    "trivial": baselines.fit_trivial,
    "popularity": baselines.fit_popularity,
    "bias": baselines.fit_bias,
}


def train(cfg: Config, names: list[str]) -> None:
    """Fit each named model. Each one saves its own predictions."""
    for name in names:
        print(f"[train] {name}")
        try:
            MODELS[name](cfg)
        except SkipModel as e:
            print(f"[train] skipped {name}: {e}")
