"""Tests for the CLI entry point (cli.py)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from athlete_vision.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_df(n_frames: int = 30) -> pd.DataFrame:
    """Minimal DataFrame mimicking PoseEstimator output."""
    n = n_frames
    data: dict = {
        "frame_index": list(range(n)),
        "timestamp_sec": [i / 30.0 for i in range(n)],
    }
    joints = [
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]
    for joint in joints:
        data[f"{joint}_x"] = [0.5] * n
        data[f"{joint}_y"] = [0.5] * n
        data[f"{joint}_z"] = [0.0] * n
        data[f"{joint}_visibility"] = [0.95] * n
    df = pd.DataFrame(data)
    df.attrs["avg_confidence"] = {j: 0.95 for j in ["left_hip", "right_hip"]}
    return df


def _pose_estimator_mock(df: pd.DataFrame) -> MagicMock:
    """Return a PoseEstimator class mock that acts as a context manager."""
    mock_cls = MagicMock()
    instance = mock_cls.return_value.__enter__.return_value
    instance.process_video.return_value = df
    return mock_cls


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

class TestProcessCommand:

    def test_missing_video_dir_shows_usage_error(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outdir:
            result = runner.invoke(main, ["process", "--output", outdir])
        assert result.exit_code != 0
        assert "Missing option '--video-dir'" in result.output

    def test_missing_output_shows_usage_error(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["process", "--video-dir", tmpdir])
        assert result.exit_code != 0
        assert "Missing option '--output'" in result.output

    def test_no_videos_exits_nonzero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            result = runner.invoke(main, [
                "process", "--video-dir", tmpdir, "--output", outdir,
            ])
        assert result.exit_code == 1
        assert "No video files found" in result.output

    @pytest.mark.parametrize("bad_complexity", ["-1", "3"])
    def test_invalid_model_complexity_shows_error(self, bad_complexity):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            result = runner.invoke(main, [
                "process",
                "--video-dir", tmpdir,
                "--output", outdir,
                "--model-complexity", bad_complexity,
            ])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    @pytest.mark.parametrize("complexity", [0, 1, 2])
    def test_valid_model_complexity_accepted(self, complexity):
        runner = CliRunner()
        fake_df = _make_fake_df()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            (Path(tmpdir) / "run.mp4").touch()
            mock_cls = _pose_estimator_mock(fake_df)
            with patch("athlete_vision.cli.PoseEstimator", mock_cls):
                result = runner.invoke(main, [
                    "process",
                    "--video-dir", tmpdir,
                    "--output", outdir,
                    "--model-complexity", str(complexity),
                ])
        assert result.exit_code == 0
        mock_cls.assert_called_once_with(model_complexity=complexity)

    def test_valid_run_produces_csv(self):
        runner = CliRunner()
        fake_df = _make_fake_df()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            (Path(tmpdir) / "athlete.mp4").touch()
            mock_cls = _pose_estimator_mock(fake_df)
            with patch("athlete_vision.cli.PoseEstimator", mock_cls):
                result = runner.invoke(main, [
                    "process",
                    "--video-dir", tmpdir,
                    "--output", outdir,
                ])
            assert result.exit_code == 0
            assert "athlete_keypoints.csv" in result.output
            assert (Path(outdir) / "athlete_keypoints.csv").exists()

    def test_pose_error_on_single_video_is_logged_and_continues(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            (Path(tmpdir) / "bad.mp4").touch()
            mock_cls = MagicMock()
            instance = mock_cls.return_value.__enter__.return_value
            instance.process_video.side_effect = ValueError("unreadable video")
            with patch("athlete_vision.cli.PoseEstimator", mock_cls):
                result = runner.invoke(main, [
                    "process",
                    "--video-dir", tmpdir,
                    "--output", outdir,
                ])
            # Error is logged but the command itself exits 0 (continues to next video)
            assert result.exit_code == 0
            assert "ERROR" in result.output

    def test_processes_only_video_extensions(self):
        """Non-video files in the directory are ignored."""
        runner = CliRunner()
        fake_df = _make_fake_df()
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            (Path(tmpdir) / "video.mp4").touch()
            (Path(tmpdir) / "notes.txt").touch()
            (Path(tmpdir) / "data.csv").touch()
            mock_cls = _pose_estimator_mock(fake_df)
            with patch("athlete_vision.cli.PoseEstimator", mock_cls):
                result = runner.invoke(main, [
                    "process",
                    "--video-dir", tmpdir,
                    "--output", outdir,
                ])
            assert result.exit_code == 0
            instance = mock_cls.return_value.__enter__.return_value
            assert instance.process_video.call_count == 1


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

class TestDownloadCommand:

    def test_no_videos_downloaded_exits_nonzero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outdir:
            with patch("athlete_vision.cli.search_and_download", return_value=([], [])):
                result = runner.invoke(main, [
                    "download", "--output-dir", outdir,
                ])
        assert result.exit_code == 1
        assert "No videos downloaded" in result.output

    def test_no_processable_videos_exits_nonzero(self):
        """Downloaded OK but output dir has no video files → exit 1."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outdir:
            # outdir remains empty of video files
            with patch("athlete_vision.cli.search_and_download", return_value=(["v1"], [])), \
                 patch("athlete_vision.cli.PoseEstimator", _pose_estimator_mock(_make_fake_df())):
                result = runner.invoke(main, [
                    "download", "--output-dir", outdir,
                ])
        assert result.exit_code == 1
        assert "No videos were successfully processed" in result.output

    def test_valid_download_writes_summary_csv(self):
        runner = CliRunner()
        fake_df = _make_fake_df()
        with tempfile.TemporaryDirectory() as outdir:
            outpath = Path(outdir)
            (outpath / "abc123_run.mp4").touch()
            metadata = {"abc123": {"title": "John Doe 40yd", "known_time": 4.32}}
            (outpath / "metadata.json").write_text(json.dumps(metadata))

            with patch("athlete_vision.cli.search_and_download", return_value=(["abc123"], [])), \
                 patch("athlete_vision.cli.PoseEstimator", _pose_estimator_mock(fake_df)):
                result = runner.invoke(main, [
                    "download", "--output-dir", outdir,
                ])
            assert result.exit_code == 0
            assert "dataset_summary.csv" in result.output
            assert (outpath / "dataset_summary.csv").exists()

    @pytest.mark.parametrize("complexity", [0, 1, 2])
    def test_model_complexity_forwarded_to_estimator(self, complexity):
        runner = CliRunner()
        fake_df = _make_fake_df()
        with tempfile.TemporaryDirectory() as outdir:
            (Path(outdir) / "vid.mp4").touch()
            mock_cls = _pose_estimator_mock(fake_df)
            with patch("athlete_vision.cli.search_and_download", return_value=(["vid"], [])), \
                 patch("athlete_vision.cli.PoseEstimator", mock_cls):
                result = runner.invoke(main, [
                    "download",
                    "--output-dir", outdir,
                    "--model-complexity", str(complexity),
                ])
        assert result.exit_code == 0
        mock_cls.assert_called_once_with(model_complexity=complexity)

    def test_failed_downloads_reported_to_stderr(self):
        runner = CliRunner()
        fake_df = _make_fake_df()
        failed = [{"video_id": "x1", "title": "Some Title", "error": "timeout"}]
        with tempfile.TemporaryDirectory() as outdir:
            (Path(outdir) / "good_video.mp4").touch()
            with patch("athlete_vision.cli.search_and_download", return_value=(["good"], failed)), \
                 patch("athlete_vision.cli.PoseEstimator", _pose_estimator_mock(fake_df)):
                result = runner.invoke(main, [
                    "download", "--output-dir", outdir,
                ])
        assert result.exit_code == 0
        assert "download(s) failed" in result.output


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

class TestBatchCommand:

    def test_missing_video_dir_shows_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["batch"])
        assert result.exit_code != 0
        assert "Missing option '--video-dir'" in result.output

    def test_file_not_found_exits_nonzero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "athlete_vision.cli.batch_process",
                side_effect=FileNotFoundError("no videos found"),
            ):
                result = runner.invoke(main, [
                    "batch", "--video-dir", tmpdir,
                ])
        assert result.exit_code == 1
        assert "no videos found" in result.output

    def test_valid_run_calls_batch_process_and_print_summary(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"], "extracted_40_time": [4.5]})
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            out_csv = Path(outdir) / "dataset.csv"
            with patch("athlete_vision.cli.batch_process", return_value=fake_df) as mock_batch, \
                 patch("athlete_vision.cli.print_summary") as mock_summary:
                result = runner.invoke(main, [
                    "batch",
                    "--video-dir", tmpdir,
                    "--output", str(out_csv),
                ])
        assert result.exit_code == 0
        mock_batch.assert_called_once()
        mock_summary.assert_called_once_with(fake_df)

    def test_output_path_appears_in_stdout(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            out_csv = Path(outdir) / "results.csv"
            with patch("athlete_vision.cli.batch_process", return_value=fake_df), \
                 patch("athlete_vision.cli.print_summary"):
                result = runner.invoke(main, [
                    "batch",
                    "--video-dir", tmpdir,
                    "--output", str(out_csv),
                ])
        assert result.exit_code == 0
        assert "results.csv" in result.output

    @pytest.mark.parametrize("complexity", [0, 1, 2])
    def test_model_complexity_forwarded(self, complexity):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.batch_process", return_value=fake_df) as mock_batch, \
                 patch("athlete_vision.cli.print_summary"):
                runner.invoke(main, [
                    "batch",
                    "--video-dir", tmpdir,
                    "--model-complexity", str(complexity),
                ])
        # batch_process(video_dir_path, output_path, model_complexity) — 3rd positional arg
        assert mock_batch.call_args[0][2] == complexity

    @pytest.mark.parametrize("bad_complexity", ["-1", "3"])
    def test_invalid_model_complexity_shows_error(self, bad_complexity):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "batch",
                "--video-dir", tmpdir,
                "--model-complexity", bad_complexity,
            ])
        assert result.exit_code != 0
        assert "Invalid value" in result.output


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

class TestWebCommand:

    @pytest.mark.parametrize("bad_port", ["1023", "65536"])
    def test_invalid_port_shows_usage_error(self, bad_port):
        runner = CliRunner()
        result = runner.invoke(main, ["web", "--port", bad_port])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    @pytest.mark.parametrize("port", [1024, 8200, 8501, 65535])
    def test_valid_port_accepted(self, port):
        runner = CliRunner()
        mock_web_app = MagicMock()
        with patch.dict("sys.modules", {"athlete_vision.web_app": mock_web_app}):
            result = runner.invoke(main, ["web", "--port", str(port)])
        assert result.exit_code == 0
        mock_web_app.run.assert_called_once_with(port)

    def test_output_mentions_localhost_and_port(self):
        runner = CliRunner()
        mock_web_app = MagicMock()
        with patch.dict("sys.modules", {"athlete_vision.web_app": mock_web_app}):
            result = runner.invoke(main, ["web", "--port", "9000"])
        assert "localhost" in result.output
        assert "9000" in result.output

    def test_default_port_is_8200(self):
        runner = CliRunner()
        mock_web_app = MagicMock()
        with patch.dict("sys.modules", {"athlete_vision.web_app": mock_web_app}):
            result = runner.invoke(main, ["web"])
        assert result.exit_code == 0
        mock_web_app.run.assert_called_once_with(8200)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

class TestPipelineCommand:

    def test_missing_video_dir_shows_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["pipeline"])
        assert result.exit_code != 0
        assert "Missing option '--video-dir'" in result.output

    def test_file_not_found_exits_nonzero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", side_effect=FileNotFoundError("no dir")), \
                 patch("athlete_vision.calibration.calibrate", return_value=1.0):
                result = runner.invoke(main, [
                    "pipeline", "--video-dir", tmpdir,
                ])
        assert result.exit_code == 1
        assert "no dir" in result.output

    def test_valid_run_calls_run_pipeline_and_print_summary(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"], "forty_time": [4.5]})
        fake_stats = {"processed": 1, "failed": 0}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)) as mock_pipe, \
                 patch("athlete_vision.cli.print_pipeline_summary") as mock_summary, \
                 patch("athlete_vision.calibration.calibrate", return_value=1.0):
                result = runner.invoke(main, [
                    "pipeline", "--video-dir", tmpdir,
                ])
        assert result.exit_code == 0
        mock_pipe.assert_called_once()
        mock_summary.assert_called_once_with(fake_df, fake_stats)

    def test_output_path_appears_in_stdout(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        fake_stats: dict = {}
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as outdir:
            out_csv = Path(outdir) / "results.csv"
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)), \
                 patch("athlete_vision.cli.print_pipeline_summary"), \
                 patch("athlete_vision.calibration.calibrate", return_value=1.0):
                result = runner.invoke(main, [
                    "pipeline",
                    "--video-dir", tmpdir,
                    "--output", str(out_csv),
                ])
        assert result.exit_code == 0
        assert "results.csv" in result.output

    @pytest.mark.parametrize("complexity", [0, 1, 2])
    def test_model_complexity_forwarded(self, complexity):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        fake_stats: dict = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)) as mock_pipe, \
                 patch("athlete_vision.cli.print_pipeline_summary"), \
                 patch("athlete_vision.calibration.calibrate", return_value=1.0):
                runner.invoke(main, [
                    "pipeline",
                    "--video-dir", tmpdir,
                    "--model-complexity", str(complexity),
                ])
        assert mock_pipe.call_args[1].get("model_complexity") == complexity

    def test_calibration_options_forwarded(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        fake_stats: dict = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)) as mock_pipe, \
                 patch("athlete_vision.cli.print_pipeline_summary"), \
                 patch("athlete_vision.calibration.calibrate", return_value=1.234) as mock_cal:
                result = runner.invoke(main, [
                    "pipeline",
                    "--video-dir", tmpdir,
                    "--calibration-distance", "40",
                    "--calibration-unit", "yards",
                ])
        assert result.exit_code == 0
        mock_cal.assert_called_once_with(calibration_distance=40.0, calibration_unit="yards")
        assert mock_pipe.call_args[1].get("calibration_factor") == pytest.approx(1.234)

    def test_calibration_echo_when_distance_provided(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        fake_stats: dict = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)), \
                 patch("athlete_vision.cli.print_pipeline_summary"), \
                 patch("athlete_vision.calibration.calibrate", return_value=0.9144):
                result = runner.invoke(main, [
                    "pipeline",
                    "--video-dir", tmpdir,
                    "--calibration-distance", "40",
                    "--calibration-unit", "yards",
                ])
        assert result.exit_code == 0
        assert "Calibration" in result.output
        assert "40" in result.output

    @pytest.mark.parametrize("bad_complexity", ["-1", "3"])
    def test_invalid_model_complexity_shows_error(self, bad_complexity):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, [
                "pipeline",
                "--video-dir", tmpdir,
                "--model-complexity", bad_complexity,
            ])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_athlete_id_forwarded(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        fake_stats: dict = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("athlete_vision.cli.run_pipeline", return_value=(fake_df, fake_stats)) as mock_pipe, \
                 patch("athlete_vision.cli.print_pipeline_summary"), \
                 patch("athlete_vision.calibration.calibrate", return_value=1.0):
                runner.invoke(main, [
                    "pipeline",
                    "--video-dir", tmpdir,
                    "--athlete-id", "john_doe",
                ])
        assert mock_pipe.call_args[1].get("athlete_id") == "john_doe"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReportCommand:

    def test_missing_csv_shows_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(main, ["report"])
        assert result.exit_code != 0
        assert "Missing option '--csv'" in result.output

    def test_empty_csv_exits_nonzero(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            # Headers-only CSV → pd.read_csv succeeds but df.empty is True
            f.write("video,extracted_40_time\n")
            csv_path = f.name
        try:
            result = runner.invoke(main, ["report", "--csv", csv_path])
        finally:
            Path(csv_path).unlink(missing_ok=True)
        assert result.exit_code == 1
        assert "empty" in result.output.lower()

    def test_default_output_path_is_html_sibling_of_csv(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"], "extracted_40_time": [4.5]})
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "batch_results.csv"
            fake_df.to_csv(csv_path, index=False)
            with patch("athlete_vision.cli.generate_html_report") as mock_report:
                result = runner.invoke(main, ["report", "--csv", str(csv_path)])
        assert result.exit_code == 0
        assert "batch_results.html" in result.output
        called_out = mock_report.call_args[0][1]
        assert Path(called_out) == csv_path.with_suffix(".html")

    def test_custom_output_path_used(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"], "extracted_40_time": [4.5]})
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            out_path = Path(tmpdir) / "my_report.html"
            fake_df.to_csv(csv_path, index=False)
            with patch("athlete_vision.cli.generate_html_report") as mock_report:
                result = runner.invoke(main, [
                    "report",
                    "--csv", str(csv_path),
                    "--output", str(out_path),
                ])
        assert result.exit_code == 0
        called_out = mock_report.call_args[0][1]
        assert Path(called_out) == out_path

    def test_generate_html_report_receives_dataframe(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"], "extracted_40_time": [4.5]})
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            fake_df.to_csv(csv_path, index=False)
            with patch("athlete_vision.cli.generate_html_report") as mock_report:
                result = runner.invoke(main, ["report", "--csv", str(csv_path)])
        assert result.exit_code == 0
        mock_report.assert_called_once()
        called_df = mock_report.call_args[0][0]
        assert list(called_df.columns) == list(fake_df.columns)

    def test_report_written_message_in_output(self):
        runner = CliRunner()
        fake_df = pd.DataFrame({"video": ["v1"]})
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            fake_df.to_csv(csv_path, index=False)
            with patch("athlete_vision.cli.generate_html_report"):
                result = runner.invoke(main, ["report", "--csv", str(csv_path)])
        assert result.exit_code == 0
        assert "Report written" in result.output
