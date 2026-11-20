from typing import Literal

from pydantic import Field

from .section import _Section


class Evaluation(_Section):
    # Negatives per user in the fixed ranking-candidate tables built by prep.
    n_negatives: int = Field(99, ge=1)
    # "uniform" (SASRec-style) or "popularity" (weighted by train counts; a harder test).
    negatives: Literal["uniform", "popularity"] = "uniform"