"""The evaluation protocol: every model is scored here, on the same split."""

from assignment_2.config import Config


def evaluate(cfg: Config, split: str = "test") -> None:
    """Score each trained model and write cfg.paths.metrics."""
    raise NotImplementedError("modules/evaluate.py: evaluate")
