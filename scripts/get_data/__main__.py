"""Ensure project datasets exist in DATA_DIR (<repo>/data by default).

Safe to re-run: datasets already present are skipped. Works the same locally and
in Codespaces, and is run by postCreateCommand (via `make data`).

    uv run python scripts/get_data.py                 # fetch the ACTIVE datasets
    uv run python scripts/get_data.py modcloth steam  # fetch specific candidates
    uv run python scripts/get_data.py --all           # fetch every candidate
    uv run python scripts/get_data.py --list          # show status of every candidate
    uv run python scripts/get_data.py --force         # re-download, even if present

Files are kept as downloaded (e.g. .json.gz stays compressed); loaders in src/ read
them directly. .zip and .tar.gz archives are extracted. Add a sha256 to a File to pin its exact bytes.
"""

import ast
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Swap for your package config once it exists, e.g.
#   from assignment_2.modules.config import config; DATA_DIR = config.paths.data_dir
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).resolve()


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
    check: Callable[[Path], str] | None = None  # quick parse; returns a one-line summary


# ---------------------------------------------------------------------------
# Sanity checks: confirm the files parse. Not the real loaders (those live in src/).
# ---------------------------------------------------------------------------


def parse_record(line: str) -> dict:
    # Many course datasets are one Python dict literal per line, not strict JSON.
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


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


# ---------------------------------------------------------------------------
# Registry: candidates from https://cseweb.ucsd.edu/~jmcauley/datasets.html that have
# direct download links. Sizes are interaction counts from that page, not file sizes.
# Citations are from memory: confirm against the page before quoting them in the writeup.
# Datasets hosted on Kaggle / Hugging Face / Google Drive folders are left out; add them
# as manual entries (no files) if the group picks one.
# ---------------------------------------------------------------------------

MCAULEY = "https://mcauleylab.ucsd.edu/public_datasets/data"

DATASETS: dict[str, Dataset] = {
    # "modcloth": Dataset(
    #     note="ModCloth clothing fit feedback (~83K transactions)",
    #     files=(File(f"{MCAULEY}/modcloth/modcloth_final_data.json.gz"),),
    #     citation="Misra, Wan & McAuley (2018). Decomposing fit semantics for product size "
    #     "recommendation in metric spaces. RecSys.",
    #     check=check_jsonl_gz,
    # ),
    # "renttherunway": Dataset(
    #     note="RentTheRunway clothing fit feedback (~193K transactions)",
    #     files=(File(f"{MCAULEY}/renttherunway/renttherunway_final_data.json.gz"),),
    #     citation="Misra, Wan & McAuley (2018). Decomposing fit semantics for product size "
    #     "recommendation in metric spaces. RecSys.",
    #     check=check_jsonl_gz,
    # ),
    # "steam": Dataset(
    #     note="Steam v1: Australian user reviews, user-item play data, bundles",
    #     files=(
    #         File(f"{MCAULEY}/steam/australian_user_reviews.json.gz"),
    #         File(f"{MCAULEY}/steam/australian_users_items.json.gz"),
    #         File(f"{MCAULEY}/steam/bundle_data.json.gz"),
    #     ),
    #     citation="Kang & McAuley (2018). Self-attentive sequential recommendation. ICDM; "
    #     "Pathak, Gupta & McAuley (2017). Generating and personalizing bundle recommendations on "
    #     "Steam. SIGIR.",
    #     check=check_jsonl_gz,
    # ),
    # "steam_v2": Dataset(
    #     note="Steam v2: ~7.8M reviews + game metadata (LARGE)",
    #     files=(
    #         File("https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"),
    #         File("https://cseweb.ucsd.edu/~wckang/steam_games.json.gz"),
    #     ),
    #     citation="Kang & McAuley (2018). Self-attentive sequential recommendation. ICDM.",
    #     check=check_jsonl_gz,
    # ),
    # "beeradvocate": Dataset(
    #     note="BeerAdvocate multi-aspect reviews (~1.6M reviews)",
    #     files=(File(f"{MCAULEY}/beer/beeradvocate.json.gz"),),
    #     citation="McAuley, Leskovec & Jurafsky (2012). Learning attitudes and attributes from "
    #     "multi-aspect reviews. ICDM.",
    #     check=check_jsonl_gz,
    # ),
    # "ratebeer": Dataset(
    #     note="RateBeer multi-aspect reviews (~2.9M reviews)",
    #     files=(File(f"{MCAULEY}/beer/ratebeer.json.gz"),),
    #     citation="McAuley, Leskovec & Jurafsky (2012). Learning attitudes and attributes from "
    #     "multi-aspect reviews. ICDM.",
    #     check=check_jsonl_gz,
    # ),
    # "google_local_2018": Dataset(
    #     note="Google Local 2018: ~11.5M reviews, places, users (LARGE)",
    #     files=(
    #         File(f"{MCAULEY}/googlelocal/places.clean.json.gz"),
    #         File(f"{MCAULEY}/googlelocal/users.clean.json.gz"),
    #         File(f"{MCAULEY}/googlelocal/reviews.clean.json.gz"),
    #     ),
    #     citation="He, Kang & McAuley (2017). Translation-based recommendation. RecSys; "
    #     "Pasricha & McAuley (2018). Translation-based factorization machines for sequential "
    #     "recommendation. RecSys.",
    #     check=check_jsonl_gz,
    # ),
    # "tradesy": Dataset(
    #     note="Tradesy peer-to-peer trades (~68K transactions)",
    #     files=(File(f"{MCAULEY}/tradesy/tradesy.json.gz"),),
    #     citation="Rappaz, Vladarean, McAuley & Catasta (2017). Bartering books to beers: "
    #     "a recommender system for exchange platforms. WSDM.",
    #     check=check_jsonl_gz,
    # ),
    # "librarything": Dataset(
    #     note="LibraryThing reviews + social links (~979K ratings)",
    #     files=(File(f"{MCAULEY}/librarything/lthing_data.tar.gz"),),
    #     citation="Zhao, McAuley & King (2014). Leveraging social connections to improve "
    #     "personalized ranking for collaborative filtering. CIKM.",
    # ),
    # "epinions": Dataset(
    #     note="Epinions reviews + trust network (~181K ratings)",
    #     files=(File(f"{MCAULEY}/epinions/epinions_data.tar.gz"),),
    #     citation="Zhao, McAuley & King (2014). Leveraging social connections to improve "
    #     "personalized ranking for collaborative filtering. CIKM.",
    # ),
    # "goodreads": Dataset(
    #     note="Goodreads: pick specific genre files from "
    #     "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html and add their URLs here",
    # ),
}

# What `make data` / postCreateCommand fetches. Set this once the group picks a dataset.
ACTIVE: list[str] = []


# ---------------------------------------------------------------------------


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
            # one flaky download shouldn't stop the others; clear partial state
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
    if not names:
        names = list(DATASETS) if flags & {"--all", "--list"} else ACTIVE

    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {', '.join(unknown)}. Options: {', '.join(DATASETS)}")

    if "--list" in flags:
        print(f"DATA_DIR = {DATA_DIR}")
        print(f"ACTIVE   = {ACTIVE or '(none yet)'}")
        for name in names:
            ds = DATASETS[name]
            status = "present" if present(name) else ("missing" if ds.files else "manual")
            print(f"  {name:18s} {status:8s} {ds.note}")
            if ds.citation:
                print(f"  {'':18s} {'':8s} cite: {ds.citation}")
        return

    if not names:
        print(
            "No ACTIVE datasets yet: set ACTIVE in scripts/get_data.py, "
            "or name candidates explicitly (see --list)"
        )
        return

    for name in names:
        fetch(name, force="--force" in flags)


if __name__ == "__main__":
    main(sys.argv[1:])