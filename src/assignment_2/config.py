"""Project paths and run settings, shared by the CLI, scripts, and notebooks.

Paths resolve from the repo root so they work the same locally and in Codespaces.
DATA_DIR / ART_DIR / RESULTS_DIR environment variables override the defaults.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default)).resolve()


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", REPO_ROOT / "data"))
    art_dir: Path = field(default_factory=lambda: _env_path("ART_DIR", REPO_ROOT / "artifacts"))
    results_dir: Path = field(
        default_factory=lambda: _env_path("RESULTS_DIR", REPO_ROOT / "results")
    )

    @property
    def metrics(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def figures(self) -> Path:
        return self.results_dir / "figures"

    def ensure(self) -> "Paths":
        # Git doesn't keep empty folders, so create the output dirs on demand.
        for d in (self.art_dir, self.results_dir, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    seed: int = 258
    dataset: str = ""  # set once the group picks one (matches a key in scripts/get_data)

    def with_paths(self, **overrides: Path | None) -> "Config":
        given = {k: Path(v).resolve() for k, v in overrides.items() if v is not None}
        return replace(self, paths=replace(self.paths, **given))


config = Config()
