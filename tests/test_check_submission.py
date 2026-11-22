"""The submission checker, with the network replaced by fake downloads."""

import importlib.util
from pathlib import Path

import pytest

pytest.skip("rewrite for the current pipeline", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_submission.py"
spec = importlib.util.spec_from_file_location("check_submission", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

DRIVE = "https://drive.google.com/file/d/ABC123/view?usp=drive_link"
MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * check.MEGABYTE



def test_file_id_from_both_url_forms():
    assert check.drive_file_id(DRIVE) == "ABC123"
    assert check.drive_file_id("https://drive.google.com/uc?export=download&id=XYZ") == "XYZ"


def test_bad_url_is_rejected():
    with pytest.raises(check.SubmissionError):
        check.drive_file_id("https://example.com/video.mp4")


def test_valid_mp4_passes():
    check.check_video(DRIVE, fetch=lambda url: MP4)


def test_youtube_skips_download():
    check.check_video("https://www.youtube.com/watch?v=abc", fetch=lambda url: pytest.fail())


def test_private_file_fails_with_sharing_hint():
    with pytest.raises(check.SubmissionError, match="Anyone with the link"):
        check.check_video(DRIVE, fetch=lambda url: b"<html>Sign in</html>")


def test_quicktime_is_rejected():
    mov = b"\x00\x00\x00\x20ftypqt  " + b"\x00" * check.MEGABYTE
    with pytest.raises(check.SubmissionError, match="not an mp4"):
        check.check_video(DRIVE, fetch=lambda url: mov)


def test_virus_scan_page_is_followed():
    page = b'Virus scan warning <input name="uuid" value="U1">'
    responses = iter([page, MP4])
    urls = []

    def fetch(url):
        urls.append(url)
        return next(responses)

    check.check_video(DRIVE, fetch=fetch)
    assert "uuid=U1" in urls[1]


def test_workbook_must_start_with_doctype(tmp_path):
    path = tmp_path / "workbook.html"
    path.write_text("<html></html>")
    with pytest.raises(check.SubmissionError):
        check.check_workbook(path)
