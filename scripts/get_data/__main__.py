"""Download the dataset named in config.toml into the configured data directory.

Safe to re-run: datasets already present are skipped. `make data` and the devcontainer
both run this. Datasets that need no download, like `synthetic`, are simply not listed.

    uv run python scripts/get_data                # fetch config.toml's dataset
    uv run python scripts/get_data modcloth       # fetch a specific entry
    uv run python scripts/get_data --list         # show the status of every entry
    uv run python scripts/get_data --force        # re-download, even if present

Files are kept as downloaded (e.g. .json.gz stays compressed); loaders in
modules/data/steam.py read them directly. .zip and .tar.gz archives are extracted.
Add a sha256 to a File to pin its exact bytes.
"""

import gzip
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from assignment_2.modules.config import get_config
from assignment_2.modules.data.records import parse_record

CONFIG = get_config()
DATA_DIR = CONFIG.paths.data_dir


@dataclass(frozen=True)
class File:
    url: str
    sha256: str | None = None  # optional: fail if the bytes don't match

    @property
    def name(self) -> str:
        return self.url.rstrip("/").split("/")[-1].split("?")[0]


@dataclass(frozen=True)
class Dataset:
    note: str
    files: tuple[File, ...] = field(default_factory=tuple)  # empty = manual download
    citation: str = ""
    # quick parse; returns a one-line summary
    check: Callable[[Path], str] | None = None


def check_jsonl_gz(dest: Path) -> str:
    # One record per line. Reports record count + first record's keys for each file.
    parts = []
    for path in sorted(dest.glob("*.json.gz")):
        n, keys = 0, None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if keys is None:
                    keys = sorted(parse_record(line))
                n += 1
        parts.append(f"{path.name}: {n:,} records, keys={keys}")
    return "; ".join(parts) or "no .json.gz files found"


# Add the group's dataset here, using the same key as its loader in dataset.
# Options and links: https://cseweb.ucsd.edu/~jmcauley/datasets.html
STEAM = "https://mcauleylab.ucsd.edu/public_datasets/data/steam"
STEAM_V2 = "https://cseweb.ucsd.edu/~wckang"
STEAM_CITATION = (
    "Kang & McAuley (2018). Self-attentive sequential recommendation. ICDM; "
    "Wan & McAuley (2018). Item recommendation on monotonic behavior chains. RecSys; "
    "Pathak, Gupta & McAuley (2017). Generating and personalizing bundle "
    "recommendations on Steam. SIGIR."
)

DATASETS: dict[str, Dataset] = {
    "steam": Dataset(
        note="Steam v1: Australian reviews + user-item play data, bundles, item metadata (~80 MB)",
        files=(
            File(f"{STEAM}/australian_user_reviews.json.gz"),  # 6.7 MB
            File(f"{STEAM}/australian_users_items.json.gz"),  # 71 MB
            File(f"{STEAM}/bundle_data.json.gz"),  # 92 KB
            # 2.7 MB, names/genres/prices
            File(f"{STEAM_V2}/steam_games.json.gz"),
        ),
        citation=STEAM_CITATION,
        check=check_jsonl_gz,
    ),
    "steam_v2": Dataset(
        note="Steam v2: full review data (1.3 GB); only fetched when named explicitly",
        files=(File(f"{STEAM_V2}/steam_reviews.json.gz"),),
        citation=STEAM_CITATION,
        check=check_jsonl_gz,
    ),
}


def present(name: str) -> bool:
    d = DATA_DIR / name
    return d.is_dir() and any(p for p in d.iterdir() if not p.name.endswith(".part"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(file: File, dest: Path) -> None:
    # Write to .part and rename at the end so an interrupted download never looks "present".
    target = dest / file.name
    part = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(file.url, part)
        if file.sha256 and sha256(part) != file.sha256:
            raise OSError(f"sha256 mismatch for {file.name}")
        if zipfile.is_zipfile(part):
            with zipfile.ZipFile(part) as zf:
                zf.extractall(dest)
        elif tarfile.is_tarfile(part):
            with tarfile.open(part) as tf:
                tf.extractall(dest, filter="data")
        else:
            part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def fetch(name: str, force: bool = False) -> None:
    ds = DATASETS[name]
    dest = DATA_DIR / name

    if present(name) and not force:
        print(f"[ok]     {name}")
    elif not ds.files:
        print(f"[manual] {name}: {ds.note}")
        return
    else:
        if force and dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[get]    {name}: {ds.note}")
        try:
            for file in ds.files:
                print(f"         {file.name}")
                download(file, dest)
            print(f"[done]   {name}")
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
            # clear partial state, so a failed download never looks "present"
            shutil.rmtree(dest, ignore_errors=True)
            print(f"[warn]   {name}: download failed ({e}); re-run this script later")
            return

    if ds.check is not None:
        try:
            print(f"[info]   {name}: {ds.check(dest)}")
        except (OSError, ValueError, KeyError, SyntaxError) as e:
            print(f"[warn]   {name}: sanity check failed ({e})")


def main(argv: list[str]) -> None:
    flags = {a for a in argv if a.startswith("--")}
    names = [a for a in argv if not a.startswith("--")]

    if "--list" in flags:
        print(f"DATA_DIR = {DATA_DIR}")
        print(f"config   = {CONFIG.dataset or '(none yet)'}")
        for name in names or DATASETS:
            ds = DATASETS[name]
            status = "present" if present(name) else ("missing" if ds.files else "manual")
            print(f"  {name:18s} {status:8s} {ds.note}")
            if ds.citation:
                print(f"  {'':18s} {'':8s} cite: {ds.citation}")
        return

    if not names:
        if CONFIG.dataset not in DATASETS:
            print(f"Nothing to download for dataset {CONFIG.dataset!r}")
            return
        names = [CONFIG.dataset]

    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {', '.join(unknown)}. Options: {', '.join(DATASETS)}")

    for name in names:
        fetch(name, force="--force" in flags)


if __name__ == "__main__":
    main(sys.argv[1:])
