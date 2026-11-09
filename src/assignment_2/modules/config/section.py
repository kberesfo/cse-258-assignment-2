from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)