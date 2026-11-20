from pydantic import Field

from .section import _Section


class Evaluation(_Section):
    k: int = Field(10, ge=1)  # cutoff for ranking metrics
    # sampled negatives per user for ranking
    n_negatives: int = Field(99, ge=1)
    # resamples for confidence intervals; 0 turns them off
    n_bootstrap: int = Field(500, ge=0)
