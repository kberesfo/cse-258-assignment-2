from pydantic import Field, model_validator

from .section import _Section


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
