from pydantic import Field, model_validator

from .section import _Section


class Split(_Section):
    valid_frac: float = Field(0.1, ge=0, lt=1)
    test_frac: float = Field(0.1, ge=0, lt=1)
    # Where a time column exists (reviews), hold out each user's latest rows; else random.
    by_time: bool = True
    # Remove held-out rows whose item never appears in train (a model can't score them).
    drop_cold: bool = True
    # Users with fewer rows stay entirely in train. >= 3 so held-out users keep a train row.
    min_per_user: int = Field(3, ge=3)

    @model_validator(mode="after")
    def _leaves_training_data(self) -> "Split":
        if self.valid_frac + self.test_frac >= 1:
            raise ValueError("valid_frac + test_frac must be < 1")
        return self
