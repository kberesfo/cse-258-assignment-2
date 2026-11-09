from pathlib import Path

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from .section import _Section


def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError(f"pyproject.toml not found above {__file__}")


REPO_ROOT = _find_repo_root()
DEFAULT_FILE = REPO_ROOT / "config.toml"


class Paths(_Section):
    data_dir: Path = REPO_ROOT / "data"
    art_dir: Path = REPO_ROOT / "artifacts"
    results_dir: Path = REPO_ROOT / "results"

    @field_validator("data_dir", "art_dir", "results_dir")
    @classmethod
    def _from_repo_root(cls, p: Path) -> Path:
        # Relative paths (in TOML or env) are relative to the repo, not the cwd.
        p = p.expanduser()
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()

    @property
    def metrics(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def figures(self) -> Path:
        return self.results_dir / "figures"

    @property
    def splits(self) -> Path:
        return self.art_dir / "splits"

    @property
    def preds(self) -> Path:
        return self.art_dir / "preds"

    def ensure(self) -> "Paths":
        # Git doesn't keep empty folders, so create the output dirs on demand.
        for d in (self.art_dir, self.splits, self.preds, self.results_dir, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self


class Split(_Section):
    valid_frac: float = Field(0.1, ge=0, lt=1)
    test_frac: float = Field(0.1, ge=0, lt=1)
    by_time: bool = True  # hold out each user's latest interactions instead of random rows
    drop_cold: bool = True  # remove held-out rows whose user or item never appears in train

    @model_validator(mode="after")
    def _leaves_training_data(self) -> "Split":
        if self.valid_frac + self.test_frac >= 1:
            raise ValueError("valid_frac + test_frac must be < 1")
        return self


class Evaluation(_Section):
    k: int = Field(10, ge=1)  # cutoff for ranking metrics
    n_negatives: int = Field(99, ge=1)  # sampled negatives per user for ranking
    n_bootstrap: int = Field(500, ge=0)  # resamples for confidence intervals; 0 turns them off
