"""Project configuration, built on pydantic-settings.

Sources, highest priority first:
  1. keyword overrides passed to load()          (the CLI uses this for --seed, --data-dir, ...)
  2. environment variables, prefixed A2_         (A2_SEED=1, A2_PATHS__DATA_DIR=/data,
                                                  A2_MODELS__BIAS__LAM=5.0)
  3. .env at the repo root                       (same names as 2; gitignored, personal)
  4. extra TOML via load(path) / A2_CONFIG       (personal experiment overrides)
  5. config.toml at the repo root                (committed: the group's agreed settings)
  6. defaults below

TOML files are deep-merged, so an override file only needs the keys it changes.
Unknown keys are errors, so typos fail loudly instead of being ignored.

Usage (CLI, notebooks, scripts):
    from assignment_2.modules.config import get_config
    cfg = get_config()
    cfg.paths.data_dir
    cfg.params("bias")           # -> {"lam": 1.0, ...} from [models.bias]

Example config.toml:
    dataset = "steam"
    seed = 258

    [split]
    valid_frac = 0.1
    test_frac = 0.1
    by_time = true

    [models.bias]
    lam = 1.0
"""

import functools
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import PrivateAttr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from .paths import DEFAULT_FILE, REPO_ROOT, Evaluation, Paths, Split


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="A2_",
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        extra="forbid",
        frozen=True,
    )

    dataset: str = ""  # a key in LOADERS (modules/core/datasets.py)
    task: Literal["rating", "binary", "ranking"] = "rating"  # selects the target and metrics
    seed: int = 258
    paths: Paths = Paths()
    split: Split = Split()
    evaluation: Evaluation = Evaluation()
    models: dict[str, dict[str, Any]] = {}  # [models.<name>] hyperparameters

    # Set by load(); not a setting.
    _toml_files: tuple[Path, ...] = PrivateAttr(default=())

    @field_validator("models", mode="after")
    @classmethod
    def _parse_env_values(cls, models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        # Env vars arrive as strings (A2_MODELS__BIAS__LAM=5.0 -> "5.0"); TOML is already typed.
        def parse(v: Any) -> Any:
            if not isinstance(v, str):
                return v
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v

        return {m: {k: parse(v) for k, v in p.items()} for m, p in models.items()}

    def params(self, model: str) -> dict[str, Any]:
        """Hyperparameters for one model ([models.<name>] in TOML); {} if none set."""
        return dict(self.models.get(model, {}))

    @property
    def sources(self) -> tuple[Path, ...]:
        """TOML files that were applied, for logging which settings produced a run."""
        return self._toml_files

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml = TomlConfigSettingsSource(settings_cls, toml_file=list(_TOML_FILES), deep_merge=True)
        return (init_settings, env_settings, dotenv_settings, toml)


# Which TOML files settings_customise_sources reads; load() sets this per call.
_TOML_FILES: tuple[Path, ...] = ()


def load(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a Config from config.toml, an optional extra TOML, env vars, and overrides.

    `path` falls back to $A2_CONFIG. Overrides with value None are ignored, so the CLI
    can pass its flags straight through, e.g. load(args.config, seed=args.seed).
    Nested overrides are partial: load(paths={"data_dir": "/tmp/d"}) keeps the other paths.
    """
    global _TOML_FILES
    files = [DEFAULT_FILE] if DEFAULT_FILE.is_file() else []
    extra = path or os.environ.get("A2_CONFIG")
    if extra:
        extra = Path(extra).expanduser().resolve()
        if not extra.is_file():
            raise FileNotFoundError(f"config file not found: {extra}")
        if extra != DEFAULT_FILE:
            files.append(extra)

    _TOML_FILES = tuple(files)
    try:
        cfg = Config(**_drop_none(overrides))
    finally:
        _TOML_FILES = ()
    cfg._toml_files = tuple(files)
    return cfg


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _drop_none(v)
            if not v:
                continue
        if v is not None:
            out[k] = v
    return out


@functools.cache
def get_config() -> Config:
    return load()
