from pydantic import Field

from .section import _Section


class Sample(_Section):
    # Fraction of users kept (whole histories). 1.0 = full data; lower it for fast dev runs.
    user_frac: float = Field(1.0, gt=0, le=1)

    @property
    def tag(self) -> str:
        """Folder name for cached outputs, so different fractions never overwrite each other."""
        return "full" if self.user_frac == 1 else f"users_{self.user_frac:g}"
