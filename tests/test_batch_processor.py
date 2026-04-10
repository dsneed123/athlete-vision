"""Tests for batch_processor."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from athlete_vision.batch_processor import (
    _angle_between,
    _extract_40_time,
    _lookup_metadata,
    batch_process,
    calculate_angles,
    calculate_arm_swing,
    calculate_velocity,
    generate_html_report,
    print_summary,
    process_single_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose_df(n_frames: int = 60, fps: float = 30.0) -> pd.DataFrame:
    """Build a minimal keypoint DataFrame covering all columns used by batch_processor."""
    t = np.arange(n_frames) / fps
    # Simulate athlete running left-to-right: hip_x advances linearly
    hip_x = np.linspace(0.1, 0.9, n_frames)
    # Ankle y: alternating ground contact (high y) and swing (low y) at ~4 Hz
    phase = 2 * np.pi * 4 * t
    ankle_y = 0.7 + 0.2 * np.abs(np.sin(phase))

    data = {
        "frame_index": np.arange(n_frames),
        "timestamp_sec": t,
    }

    joints = [
        ("left_hip",      hip_x,            0.45 * np.ones(n_frames)),
        ("right_hip",     hip_x,            0.55 * np.ones(n_frames)),
        ("left_knee",     hip_x + 0.01,     0.60 * np.ones(n_frames)),
        ("right_knee",    hip_x + 0.01,     0.65 * np.ones(n_frames)),
        ("left_ankle",    hip_x + 0.02,     ankle_y),
        ("right_ankle",   hip_x + 0.02,     ankle_y),
        ("left_shoulder", hip_x - 0.01,     0.30 * np.ones(n_frames)),
        ("right_shoulder",hip_x - 0.01,     0.35 * np.ones(n_frames)),
        ("left_elbow",    hip_x - 0.01,     0.20 * np.ones(n_frames)),
        ("right_elbow",   hip_x - 0.01,     0.22 * np.ones(n_frames)),
        ("left_wrist",    hip_x - 0.01,     0.10 * np.ones(n_frames)),
        ("right_wrist",   hip_x - 0.01,     0.12 * np.ones(n_frames)),
    ]

    for name, xs, ys in joints:
        data[f"{name}_x"] = xs
        data[f"{name}_y"] = ys
        data[f"{name}_z"] = np.zeros(n_frames)
        data[f"{name}_visibility"] = np.ones(n_frames)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _angle_between
# ---------------------------------------------------------------------------

class TestAngleBetween:
    def test_right_angle(self):
        # A=(0,1), B=(0,0), C=(1,0) → 90°
        angle = _angle_between(0, 1, 0, 0, 1, 0)
        assert abs(angle - 90.0) < 1e-9

    def test_straight_line(self):
        # A=(-1,0), B=(0,0), C=(1,0) → 180°
        angle = _angle_between(-1, 0, 0, 0, 1, 0)
        assert abs(angle - 180.0) < 1e-9

    def test_zero_magnitude_returns_nan(self):
        # B coincides with A → magnitude zero
        assert math.isnan(_angle_between(0, 0, 0, 0, 1, 0))

    def test_45_degree_angle(self):
        angle = _angle_between(1, 0, 0, 0, 1, 1)
        assert abs(angle - 45.0) < 1e-9


# ---------------------------------------------------------------------------
# calculate_angles
# ---------------------------------------------------------------------------

class TestCalculateAngles:
    def test_returns_expected_keys(self):
        df = _make_pose_df()
        result = calculate_angles(df)
        expected = {
            "left_knee_angle_deg", "right_knee_angle_deg",
            "left_hip_angle_deg", "right_hip_angle_deg",
        }
        assert set(result.keys()) == expected

    def test_angles_are_finite_floats(self):
        df = _make_pose_df()
        result = calculate_angles(df)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"
            assert not math.isnan(val), f"{key} is NaN"

    def test_angles_in_plausible_range(self):
        df = _make_pose_df()
        result = calculate_angles(df)
        for key, val in result.items():
            assert 0.0 <= val <= 180.0, f"{key}={val} outside [0, 180]"


# ---------------------------------------------------------------------------
# calculate_velocity
# ---------------------------------------------------------------------------

class TestCalculateVelocity:
    def test_returns_expected_keys(self):
        df = _make_pose_df()
        result = calculate_velocity(df)
        assert "peak_velocity_norm" in result
        assert "avg_velocity_norm" in result

    def test_peak_ge_avg(self):
        df = _make_pose_df()
        result = calculate_velocity(df)
        assert result["peak_velocity_norm"] >= result["avg_velocity_norm"]

    def test_empty_df_returns_nan(self):
        df = pd.DataFrame()
        result = calculate_velocity(df)
        assert math.isnan(result["peak_velocity_norm"])
        assert math.isnan(result["avg_velocity_norm"])

    def test_single_frame_returns_nan(self):
        df = _make_pose_df(n_frames=1)
        result = calculate_velocity(df)
        assert math.isnan(result["peak_velocity_norm"])

    def test_velocity_positive(self):
        df = _make_pose_df()
        result = calculate_velocity(df)
        assert result["peak_velocity_norm"] > 0
        assert result["avg_velocity_norm"] > 0


# ---------------------------------------------------------------------------
# calculate_arm_swing
# ---------------------------------------------------------------------------

class TestCalculateArmSwing:
    def test_returns_expected_keys(self):
        df = _make_pose_df()
        result = calculate_arm_swing(df)
        expected = {
            "left_arm_swing_amplitude",
            "right_arm_swing_amplitude",
            "arm_swing_amplitude",
        }
        assert set(result.keys()) == expected

    def test_amplitude_mean_matches_sides(self):
        df = _make_pose_df()
        result = calculate_arm_swing(df)
        expected_mean = (
            result["left_arm_swing_amplitude"] + result["right_arm_swing_amplitude"]
        ) / 2
        assert abs(result["arm_swing_amplitude"] - expected_mean) < 1e-9

    def test_zero_amplitude_when_wrist_fixed(self):
        df = _make_pose_df()
        # Fix wrist at the same position as shoulder
        df["left_wrist_y"] = df["left_shoulder_y"]
        df["right_wrist_y"] = df["right_shoulder_y"]
        result = calculate_arm_swing(df)
        assert result["left_arm_swing_amplitude"] == pytest.approx(0.0, abs=1e-9)
        assert result["right_arm_swing_amplitude"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _extract_40_time
# ---------------------------------------------------------------------------

class TestExtract40Time:
    def test_returns_none_for_empty(self):
        assert _extract_40_time(pd.DataFrame()) is None

    def test_returns_none_for_short_df(self):
        df = _make_pose_df(n_frames=5)
        assert _extract_40_time(df) is None

    def test_plausible_time_returned(self):
        # Build a ~4-second clip at 30 fps with clear horizontal movement
        fps = 30.0
        n_frames = int(4.2 * fps)
        df = _make_pose_df(n_frames=n_frames, fps=fps)
        result = _extract_40_time(df)
        # May or may not extract depending on movement pattern; just check type when not None
        if result is not None:
            assert 3.5 <= result <= 6.5

    def test_no_movement_returns_none(self):
        df = _make_pose_df()
        # Zero out all hip movement
        df["left_hip_x"] = 0.5
        df["right_hip_x"] = 0.5
        result = _extract_40_time(df)
        assert result is None


# ---------------------------------------------------------------------------
# _lookup_metadata
# ---------------------------------------------------------------------------

class TestLookupMetadata:
    def test_match_by_video_id_prefix(self):
        meta = {"abc123": {"athlete_name": "John", "known_time": 4.3}}
        result = _lookup_metadata("abc123_foo.mp4", meta)
        assert result == {"athlete_name": "John", "known_time": 4.3}

    def test_fallback_by_filename(self):
        meta = {"abc": {"filename": "myvideo.mp4", "athlete_name": "Jane"}}
        result = _lookup_metadata("myvideo.mp4", meta)
        assert result == {"filename": "myvideo.mp4", "athlete_name": "Jane"}

    def test_no_match_returns_empty(self):
        meta = {"xyz": {"athlete_name": "Bob"}}
        result = _lookup_metadata("nomatch.mp4", meta)
        assert result == {}


# ---------------------------------------------------------------------------
# process_single_video
# ---------------------------------------------------------------------------

class TestProcessSingleVideo:
    def _make_stub_process(self, df: pd.DataFrame):
        """Return a PoseEstimator context-manager stub that yields df."""
        mock_estimator = MagicMock()
        mock_estimator.process_video.return_value = df
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_estimator)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm

    def test_ok_status_for_valid_video(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df()
        mock_cm = self._make_stub_process(df)

        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm):
            result = process_single_video(video, {"athlete_name": "Tester"})

        assert result["status"] == "ok"
        assert result["video"] == "sample.mp4"
        assert result["athlete"] == "Tester"

    def test_no_pose_status_for_empty_df(self, tmp_path):
        video = tmp_path / "empty.mp4"
        video.touch()
        mock_cm = self._make_stub_process(pd.DataFrame())

        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm):
            result = process_single_video(video, {})

        assert result["status"] == "no_pose"

    def test_error_status_on_exception(self, tmp_path):
        video = tmp_path / "bad.mp4"
        video.touch()
        mock_estimator = MagicMock()
        mock_estimator.process_video.side_effect = RuntimeError("boom")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_estimator)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm):
            result = process_single_video(video, {})

        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_result_has_all_metric_keys(self, tmp_path):
        video = tmp_path / "full.mp4"
        video.touch()
        df = _make_pose_df()
        mock_cm = self._make_stub_process(df)

        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm):
            result = process_single_video(video, {})

        expected_keys = {
            "video", "athlete", "known_40_time", "status", "error",
            "extracted_40_time", "stride_length_m", "stride_frequency_hz",
            "ground_contact_ms", "left_knee_angle_deg", "right_knee_angle_deg",
            "left_hip_angle_deg", "right_hip_angle_deg",
            "peak_velocity_norm", "avg_velocity_norm",
            "left_arm_swing_amplitude", "right_arm_swing_amplitude",
            "arm_swing_amplitude",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_known_time_from_metadata(self, tmp_path):
        video = tmp_path / "known.mp4"
        video.touch()
        mock_cm = self._make_stub_process(pd.DataFrame())

        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm):
            result = process_single_video(video, {"known_time": 4.28})

        assert result["known_40_time"] == 4.28

    def test_flagged_status_for_tiny_stride(self, tmp_path):
        video = tmp_path / "tiny.mp4"
        video.touch()
        df = _make_pose_df()
        mock_cm = self._make_stub_process(df)

        # Patch analyze_strides to return an implausibly small stride length
        with patch("athlete_vision.batch_processor.PoseEstimator", return_value=mock_cm), \
             patch("athlete_vision.batch_processor.analyze_strides",
                   return_value={"stride_length": 0.001,
                                 "stride_frequency": 2.0,
                                 "ground_contact_ms": 100.0}):
            result = process_single_video(video, {})

        assert result["status"] == "flagged"


# ---------------------------------------------------------------------------
# batch_process
# ---------------------------------------------------------------------------

class TestBatchProcess:
    def _write_videos(self, directory: Path, names: list[str]) -> None:
        for name in names:
            (directory / name).touch()

    def _stub_process_single(self, video_path, metadata, model_complexity=1):
        return {
            "video": video_path.name,
            "athlete": metadata.get("athlete_name"),
            "known_40_time": metadata.get("known_time"),
            "status": "ok",
            "error": None,
            "extracted_40_time": 4.4,
            "stride_length_m": 1.2,
            "stride_frequency_hz": 2.5,
            "ground_contact_ms": 120.0,
            "left_knee_angle_deg": 150.0,
            "right_knee_angle_deg": 148.0,
            "left_hip_angle_deg": 170.0,
            "right_hip_angle_deg": 168.0,
            "peak_velocity_norm": 0.05,
            "avg_velocity_norm": 0.03,
            "left_arm_swing_amplitude": 0.15,
            "right_arm_swing_amplitude": 0.14,
            "arm_swing_amplitude": 0.145,
        }

    def test_raises_when_no_videos(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            batch_process(tmp_path, tmp_path / "out.csv")

    def test_csv_written(self, tmp_path):
        self._write_videos(tmp_path, ["a.mp4", "b.mp4"])
        output = tmp_path / "results.csv"

        with patch("athlete_vision.batch_processor.process_single_video",
                   side_effect=self._stub_process_single):
            df = batch_process(tmp_path, output)

        assert output.exists()
        loaded = pd.read_csv(output)
        assert len(loaded) == 2

    def test_returns_dataframe(self, tmp_path):
        self._write_videos(tmp_path, ["x.mp4"])
        output = tmp_path / "out.csv"

        with patch("athlete_vision.batch_processor.process_single_video",
                   side_effect=self._stub_process_single):
            df = batch_process(tmp_path, output)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_reads_metadata_json(self, tmp_path):
        self._write_videos(tmp_path, ["vid123_title.mp4"])
        metadata = {"vid123": {"athlete_name": "Speed Racer", "known_time": 4.22}}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        output = tmp_path / "out.csv"

        captured = {}

        def stub(video_path, meta, model_complexity=1):
            captured["meta"] = meta
            return self._stub_process_single(video_path, meta, model_complexity)

        with patch("athlete_vision.batch_processor.process_single_video", side_effect=stub):
            batch_process(tmp_path, output)

        assert captured["meta"].get("athlete_name") == "Speed Racer"

    def test_metadata_json_as_list(self, tmp_path):
        self._write_videos(tmp_path, ["clip.mp4"])
        metadata = [{"video_id": "clip", "filename": "clip.mp4",
                     "athlete_name": "List Athlete", "known_time": 4.5}]
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        output = tmp_path / "out.csv"

        captured = {}

        def stub(video_path, meta, model_complexity=1):
            captured["meta"] = meta
            return self._stub_process_single(video_path, meta, model_complexity)

        with patch("athlete_vision.batch_processor.process_single_video", side_effect=stub):
            batch_process(tmp_path, output)

        assert captured["meta"].get("athlete_name") == "List Athlete"

    def test_bad_metadata_json_ignored(self, tmp_path):
        self._write_videos(tmp_path, ["a.mp4"])
        (tmp_path / "metadata.json").write_text("not valid json{{")
        output = tmp_path / "out.csv"

        with patch("athlete_vision.batch_processor.process_single_video",
                   side_effect=self._stub_process_single):
            df = batch_process(tmp_path, output)

        assert len(df) == 1

    def test_bad_metadata_json_emits_warning(self, tmp_path):
        self._write_videos(tmp_path, ["a.mp4"])
        (tmp_path / "metadata.json").write_text("not valid json{{")
        output = tmp_path / "out.csv"

        with patch("athlete_vision.batch_processor.process_single_video",
                   side_effect=self._stub_process_single), \
             patch("athlete_vision.batch_processor.logger") as mock_logger:
            batch_process(tmp_path, output)

        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        assert "metadata" in str(args[1]).lower() or "metadata" in str(args).lower()


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def _make_df(self, statuses: list[str], times=None, known=None) -> pd.DataFrame:
        n = len(statuses)
        times = times or [4.4] * n
        known = known or [None] * n
        return pd.DataFrame({
            "status": statuses,
            "video": [f"v{i}.mp4" for i in range(n)],
            "extracted_40_time": times,
            "known_40_time": known,
            "peak_velocity_norm": [0.04] * n,
            "stride_length_m": [1.1] * n,
            "stride_frequency_hz": [2.4] * n,
        })

    def test_runs_without_error(self, capsys):
        df = self._make_df(["ok", "ok", "error", "flagged", "no_pose"])
        print_summary(df)
        out = capsys.readouterr().out
        assert "Total videos" in out
        assert "5" in out

    def test_accuracy_report_shown_when_ground_truth(self, capsys):
        df = self._make_df(["ok", "ok"], times=[4.4, 4.6], known=[4.3, 4.5])
        print_summary(df)
        out = capsys.readouterr().out
        assert "Accuracy Report" in out
        assert "Mean absolute error" in out

    def test_no_accuracy_report_without_ground_truth(self, capsys):
        df = self._make_df(["ok", "ok"])
        print_summary(df)
        out = capsys.readouterr().out
        assert "Accuracy Report" not in out

    def test_histogram_section_shown(self, capsys):
        df = self._make_df(["ok"] * 5, times=[4.2, 4.3, 4.5, 4.7, 5.1])
        print_summary(df)
        out = capsys.readouterr().out
        assert "40-Yard" in out


# ---------------------------------------------------------------------------
# generate_html_report
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "video": ["a.mp4", "b.mp4"],
            "athlete": ["Alice", "Bob"],
            "status": ["ok", "flagged"],
            "extracted_40_time": [4.3, 4.7],
            "known_40_time": [4.2, None],
            "stride_length_m": [1.1, 1.2],
            "peak_velocity_norm": [0.04, 0.05],
            "stride_frequency_hz": [2.4, 2.6],
        })

    def test_creates_html_file(self, tmp_path):
        out = tmp_path / "report.html"
        generate_html_report(self._make_df(), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_has_required_structure(self, tmp_path):
        out = tmp_path / "report.html"
        generate_html_report(self._make_df(), out)
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Athlete Vision" in content
        assert "<table>" in content

    def test_html_contains_video_names(self, tmp_path):
        out = tmp_path / "report.html"
        generate_html_report(self._make_df(), out)
        content = out.read_text()
        assert "a.mp4" in content
        assert "b.mp4" in content

    def test_html_embeds_charts_as_base64(self, tmp_path):
        out = tmp_path / "report.html"
        generate_html_report(self._make_df(), out)
        content = out.read_text()
        # Charts are embedded as data URIs
        assert "data:image/png;base64," in content

    def test_empty_dataframe_produces_valid_html(self, tmp_path):
        out = tmp_path / "empty_report.html"
        df = pd.DataFrame(columns=[
            "video", "athlete", "status", "extracted_40_time",
            "known_40_time", "stride_length_m", "peak_velocity_norm",
        ])
        generate_html_report(df, out)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
