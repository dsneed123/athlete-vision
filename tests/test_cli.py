"""Tests for CLI command behaviour."""

import pandas as pd
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from athlete_vision.cli import main


def test_download_cmd_prints_failed_summary(tmp_path):
    """download command prints a failure summary to stderr on partial failure.

    Simulates a yt-dlp download failure through the mocked search_and_download
    return value, then asserts:
      - the failed list is non-empty
      - the CLI echoes each failure's video_id, title, and error to stderr
    """
    # Create a dummy video file so the pose extraction loop has something to process
    (tmp_path / "vid1_Good_Athlete_4.38_40_yard_dash.mp4").touch()

    downloaded = [
        {
            "video_id": "vid1",
            "title": "Good Athlete 4.38 40 yard dash",
            "athlete_name": "Good Athlete",
            "known_time": 4.38,
            "url": "https://youtube.com/watch?v=vid1",
            "duration": 10,
            "query": "40 yard dash",
        }
    ]
    failed = [
        {
            "video_id": "vid2",
            "title": "Bad Video that failed to download",
            "url": "https://youtube.com/watch?v=vid2",
            "error": "network timeout",
        }
    ]

    mock_df = pd.DataFrame({"x": [1, 2, 3]})

    with (
        patch("athlete_vision.cli.search_and_download", return_value=(downloaded, failed)),
        patch("athlete_vision.cli.PoseEstimator") as mock_estimator_cls,
    ):
        mock_estimator = MagicMock()
        mock_estimator.__enter__ = lambda s: s
        mock_estimator.__exit__ = MagicMock(return_value=False)
        mock_estimator.process_video.return_value = mock_df
        mock_estimator_cls.return_value = mock_estimator

        runner = CliRunner()
        result = runner.invoke(
            main, ["download", "--output-dir", str(tmp_path), "--count", "2"]
        )

    assert len(failed) == 1
    assert "1 download(s) failed" in result.stderr
    assert "vid2" in result.stderr
    assert "network timeout" in result.stderr
