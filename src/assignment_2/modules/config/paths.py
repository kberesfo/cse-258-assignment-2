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







