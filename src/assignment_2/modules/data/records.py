"""Read gzipped one-record-per-line files (strict JSON or Python dict literals).

Works for any McAuley-style dataset. `keep_user` filters lines by user before parsing,
which is where the speedup for sampled dev runs comes from.
"""

import ast
import gzip
import json
import re
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

_USER_ID = re.compile(r"""['"]user_id['"]:\s*['"]([^'"]+)['"]""")


def user_sampler(frac: float, seed: int) -> Callable[[str], bool] | None:
    """Deterministic keep(user_id) for a fraction of users; None means keep everyone.

    Hash-based, so the same users are kept in every file, on every machine, every run,
    and a larger frac is a superset of a smaller one (same seed).
    """
    if not 0 < frac <= 1:
        raise ValueError(f"user_frac must be in (0, 1], got {frac}")
    if frac == 1:
        return None
    cutoff = int(frac * 2**32)
    return lambda uid: zlib.crc32(f"{seed}:{uid}".encode()) < cutoff


def parse_record(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def records(path: Path, keep_user: Callable[[str], bool] | None = None) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if keep_user is None:
                yield parse_record(line)
                continue
            # Cheap regex on the start of the raw line, so skipped users are never parsed.
            # If user_id isn't near the start, parse the line and check the real field.
            m = _USER_ID.search(line, 0, 200)
            if m is not None:
                if keep_user(m.group(1)):
                    yield parse_record(line)
            else:
                rec = parse_record(line)
                if keep_user(str(rec.get("user_id"))):
                    yield rec
