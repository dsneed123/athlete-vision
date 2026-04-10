"""Tests for video_downloader."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from athlete_vision.video_downloader import (
    _extract_metadata,
    _sanitize_filename,
    search_and_download,
)


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------

def test_sanitize_filename_strips_unsafe_chars():
    result = _sanitize_filename("John Smith <40yd>: 4.38!")
    # Spaces are preserved; angle brackets, colon, and bang are replaced with _
    assert "<" not in result
    assert ">" not in result
    assert "!" not in result
    assert "4.38" in result


def test_sanitize_filename_truncates():
    assert len(_sanitize_filename("a" * 200)) == 80


# ---------------------------------------------------------------------------
# _extract_metadata
# ---------------------------------------------------------------------------

def test_extract_metadata_parses_time_from_title():
    meta = _extract_metadata("John Smith 4.38 40 yard dash")
    assert meta["known_time"] == pytest.approx(4.38)


def test_extract_metadata_falls_back_to_description():
    meta = _extract_metadata("40 yard dash highlight", description="He ran a 4.45")
    assert meta["known_time"] == pytest.approx(4.45)


def test_extract_metadata_ignores_out_of_range_times():
    meta = _extract_metadata("40 yard dash 6.00 seconds")
    assert meta["known_time"] is None


def test_extract_metadata_parses_athlete_name():
    meta = _extract_metadata("James Brown 4.38 forty")
    assert meta["athlete_name"] == "James Brown"


def test_extract_metadata_no_name_when_lowercase():
    meta = _extract_metadata("awesome forty yard dash run 4.50")
    assert meta["athlete_name"] is None


# ---------------------------------------------------------------------------
# search_and_download — return type and failure tracking
# ---------------------------------------------------------------------------

def _make_flat_entry(video_id: str, title: str, duration: int = 10) -> dict:
    return {"id": video_id, "title": title, "duration": duration}


@patch("athlete_vision.video_downloader.yt_dlp.YoutubeDL")
def test_search_and_download_returns_tuple(mock_ydl_cls, tmp_path):
    """Return value must be a (downloaded, failed_downloads) tuple."""
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = lambda s: s
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = {
        "entries": [_make_flat_entry("vid1", "John Doe 4.38 40 yard dash")]
    }
    mock_ydl.download.return_value = None
    mock_ydl_cls.return_value = mock_ydl

    result = search_and_download(["40 yard dash"], count=1, output_dir=tmp_path)
    assert isinstance(result, tuple)
    assert len(result) == 2
    downloaded, failed = result
    assert isinstance(downloaded, list)
    assert isinstance(failed, list)


@patch("athlete_vision.video_downloader.yt_dlp.YoutubeDL")
def test_download_failure_appended_to_failed_list(mock_ydl_cls, tmp_path):
    """When yt_dlp.download() raises, the video must appear in failed_downloads."""
    search_ydl = MagicMock()
    search_ydl.__enter__ = lambda s: s
    search_ydl.__exit__ = MagicMock(return_value=False)
    search_ydl.extract_info.return_value = {
        "entries": [_make_flat_entry("vid2", "Broken Video 4.50 dash")]
    }

    download_ydl = MagicMock()
    download_ydl.__enter__ = lambda s: s
    download_ydl.__exit__ = MagicMock(return_value=False)
    download_ydl.download.side_effect = RuntimeError("network timeout")

    # First call → search, second call → download
    mock_ydl_cls.side_effect = [search_ydl, download_ydl]

    downloaded, failed = search_and_download(["40 yard dash"], count=1, output_dir=tmp_path)

    assert downloaded == []
    assert len(failed) == 1
    assert failed[0]["video_id"] == "vid2"
    assert "network timeout" in failed[0]["error"]
    assert "url" in failed[0]
    assert "title" in failed[0]


@patch("athlete_vision.video_downloader.yt_dlp.YoutubeDL")
def test_metadata_fetch_failure_logs_warning(mock_ydl_cls, tmp_path, caplog):
    """A metadata fetch failure must emit a WARNING log (not silently pass)."""
    # First call (search): returns an entry with no duration
    search_ydl = MagicMock()
    search_ydl.__enter__ = lambda s: s
    search_ydl.__exit__ = MagicMock(return_value=False)
    entry_no_duration = {"id": "vid3", "title": "Some Athlete 4.60", "duration": None}
    search_ydl.extract_info.return_value = {"entries": [entry_no_duration]}

    # Second call (full-info fetch): raises
    info_ydl = MagicMock()
    info_ydl.__enter__ = lambda s: s
    info_ydl.__exit__ = MagicMock(return_value=False)
    info_ydl.extract_info.side_effect = RuntimeError("info fetch failed")

    # Third call (download): succeeds
    dl_ydl = MagicMock()
    dl_ydl.__enter__ = lambda s: s
    dl_ydl.__exit__ = MagicMock(return_value=False)
    dl_ydl.download.return_value = None

    mock_ydl_cls.side_effect = [search_ydl, info_ydl, dl_ydl]

    with caplog.at_level(logging.WARNING, logger="athlete_vision.video_downloader"):
        downloaded, failed = search_and_download(["40 yard dash"], count=1, output_dir=tmp_path)

    assert any("vid3" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
