import ast
import gzip
import json
from collections.abc import Iterator
from pathlib import Path


def _records(path: Path) -> Iterator[dict]:
    # One record per line; these files are Python dict literals, not strict JSON.
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)
