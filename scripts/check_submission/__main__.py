"""Check submission files the way the spec's autograder does.

    uv run python scripts/check_submission.py [submit_dir]

The mp4 test reads the file's header directly instead of using libmagic, so teammates
don't need a system library. It accepts the common mp4 container brands.
"""

import sys
import urllib.request
from pathlib import Path

MEGABYTE = 1024 * 1024
NON_MP4_BRANDS = {b"qt  "}  # QuickTime files also have an ftyp header


class SubmissionError(Exception):
    pass


def check_workbook(path: Path) -> None:
    if not path.is_file():
        raise SubmissionError(f"missing {path}; run `make workbook`")
    if not path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>"):
        raise SubmissionError(f"{path} does not start with <!DOCTYPE html>")


def read_video_url(path: Path) -> str:
    if not path.is_file():
        raise SubmissionError(f"missing {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise SubmissionError(f"{path} must contain exactly one line, found {len(lines)}")
    return lines[0].strip()


def drive_file_id(url: str) -> str:
    """Extract the Drive file ID the same way the autograder does."""
    if "id=" in url:
        return url.split("id=")[1].split("/")[0]
    if "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    raise SubmissionError(f"couldn't find a Google Drive file ID in {url}")


def is_mp4(data: bytes) -> bool:
    return data[4:8] == b"ftyp" and data[8:12] not in NON_MP4_BRANDS


def first_megabyte(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read(MEGABYTE)


def check_video(url: str, fetch=first_megabyte) -> None:
    if "youtube" in url or "youtu.be" in url:
        print("check: YouTube link; confirm by hand that it is publicly viewable")
        return

    file_id = drive_file_id(url)
    chunk = fetch(f"https://drive.google.com/uc?export=download&id={file_id}")
    if len(chunk) < MEGABYTE:
        page = chunk.decode("utf-8", errors="replace")
        if "Virus scan warning" not in page:
            raise SubmissionError(
                "the download is under 1 MB, so it probably isn't a video. "
                'Check that sharing is set to "Anyone with the link".'
            )
        uuid = page.split('uuid" value="')[1].split('"')[0]
        chunk = fetch(
            f"https://drive.usercontent.google.com/download?id={file_id}"
            f"&export=download&confirm=t&uuid={uuid}"
        )
    if not is_mp4(chunk):
        raise SubmissionError("the downloaded file is not an mp4")


def main(argv: list[str]) -> None:
    submit = Path(argv[0] if argv else "submit")
    try:
        check_workbook(submit / "workbook.html")
        print("check: workbook.html is html")
        check_video(read_video_url(submit / "video_url.txt"))
        print("check: video_url.txt passes")
    except SubmissionError as e:
        sys.exit(f"check: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
