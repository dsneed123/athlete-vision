"""Tests for the full analysis pipeline (pipeline.py)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from athlete_vision.pipeline import (
    _check_data_quality,
    _get_video_aspect_ratio,
    _OUTPUT_COLUMNS,
    print_pipeline_summary,
    process_video,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose_df(
    n_frames: int = 120,
    fps: float = 30.0,
    visibility: float = 0.95,
) -> pd.DataFrame:
    """Build a synthetic pose DataFrame that passes all quality checks."""
    t = np.arange(n_frames) / fps
    hip_x = np.linspace(0.1, 0.9, n_frames)
    phase = 2 * np.pi * 4 * t
    ankle_y = 0.7 + 0.2 * np.abs(np.sin(phase))

    data: dict = {
        "frame_index": np.arange(n_frames),
        "timestamp_sec": t,
    }

    joints = [
        ("left_hip",       hip_x,        0.45 * np.ones(n_frames)),
        ("right_hip",      hip_x,        0.55 * np.ones(n_frames)),
        ("left_knee",      hip_x + 0.01, 0.60 * np.ones(n_frames)),
        ("right_knee",     hip_x + 0.01, 0.65 * np.ones(n_frames)),
        ("left_ankle",     hip_x + 0.02, ankle_y),
        ("right_ankle",    hip_x + 0.02, ankle_y),
        ("left_shoulder",  hip_x - 0.01, 0.30 * np.ones(n_frames)),
        ("right_shoulder", hip_x - 0.01, 0.35 * np.ones(n_frames)),
        ("left_elbow",     hip_x - 0.01, 0.20 * np.ones(n_frames)),
        ("right_elbow",    hip_x - 0.01, 0.22 * np.ones(n_frames)),
        ("left_wrist",     hip_x - 0.01, 0.10 * np.ones(n_frames)),
        ("right_wrist",    hip_x - 0.01, 0.12 * np.ones(n_frames)),
    ]

    for name, xs, ys in joints:
        data[f"{name}_x"] = xs
        data[f"{name}_y"] = ys
        data[f"{name}_z"] = np.zeros(n_frames)
        data[f"{name}_visibility"] = np.full(n_frames, visibility)

    df = pd.DataFrame(data)
    # Populate attrs as PoseEstimator would
    critical = [
        "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    ]
    df.attrs["avg_confidence"] = {j: visibility for j in critical}
    return df


def _make_pose_estimator_stub(df: pd.DataFrame) -> MagicMock:
    """Return a PoseEstimator context-manager mock that yields df."""
    estimator = MagicMock()
    estimator.process_video.return_value = df
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=estimator)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# _get_video_aspect_ratio
# ---------------------------------------------------------------------------

class TestGetVideoAspectRatio:
    def test_returns_none_when_video_unopenable(self, tmp_path):
        fake = tmp_path / "nonexistent.mp4"
        # File does not exist — cv2.VideoCapture will fail to open
        result = _get_video_aspect_ratio(fake)
        assert result is None

    def test_returns_float_for_valid_video(self, tmp_path):
        # Patch cv2.VideoCapture to simulate a 1920x1080 video
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: (
            1920.0 if prop == 3 else 1080.0  # CAP_PROP_FRAME_WIDTH=3, HEIGHT=4
        )
        with patch("athlete_vision.pipeline.cv2.VideoCapture", return_value=cap_mock):
            ratio = _get_video_aspect_ratio(tmp_path / "video.mp4")
        assert ratio == pytest.approx(1920 / 1080, rel=1e-3)


# ---------------------------------------------------------------------------
# _check_data_quality
# ---------------------------------------------------------------------------

class TestCheckDataQuality:
    def _make_video(self, tmp_path) -> Path:
        v = tmp_path / "video.mp4"
        v.touch()
        return v

    def test_ok_when_all_criteria_pass(self, tmp_path):
        df = _make_pose_df(n_frames=120)
        video = self._make_video(tmp_path)
        # Patch aspect ratio to standard 16:9
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=16 / 9):
            result = _check_data_quality(df, video)
        assert result == "OK"

    def test_review_when_too_few_frames(self, tmp_path):
        df = _make_pose_df(n_frames=50)
        video = self._make_video(tmp_path)
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=16 / 9):
            result = _check_data_quality(df, video)
        assert result == "REVIEW"

    def test_review_when_high_tracking_loss(self, tmp_path):
        df = _make_pose_df(n_frames=120)
        # Set >10 % of frames to NaN for a critical joint
        n_nan = int(0.15 * len(df))
        df.loc[:n_nan, "left_hip_x"] = float("nan")
        video = self._make_video(tmp_path)
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=16 / 9):
            result = _check_data_quality(df, video)
        assert result == "REVIEW"

    def test_review_when_low_confidence(self, tmp_path):
        df = _make_pose_df(n_frames=120, visibility=0.7)
        # Overwrite attrs to reflect low confidence
        df.attrs["avg_confidence"] = {
            j: 0.7 for j in
            ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        }
        video = self._make_video(tmp_path)
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=16 / 9):
            result = _check_data_quality(df, video)
        assert result == "REVIEW"

    def test_review_when_non_standard_aspect_ratio(self, tmp_path):
        df = _make_pose_df(n_frames=120)
        video = self._make_video(tmp_path)
        # 9:16 portrait ratio — should be flagged
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=9 / 16):
            result = _check_data_quality(df, video)
        assert result == "REVIEW"

    def test_ok_when_aspect_ratio_unavailable(self, tmp_path):
        """When aspect ratio cannot be determined, skip the check (don't penalise)."""
        df = _make_pose_df(n_frames=120)
        video = self._make_video(tmp_path)
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=None):
            result = _check_data_quality(df, video)
        assert result == "OK"

    def test_review_exact_frame_boundary(self, tmp_path):
        """Exactly 99 frames → REVIEW; exactly 100 frames → not REVIEW on frames alone."""
        video = self._make_video(tmp_path)
        with patch("athlete_vision.pipeline._get_video_aspect_ratio", return_value=16 / 9):
            df_99 = _make_pose_df(n_frames=99)
            assert _check_data_quality(df_99, video) == "REVIEW"
            df_100 = _make_pose_df(n_frames=100)
            assert _check_data_quality(df_100, video) == "OK"


# ---------------------------------------------------------------------------
# process_video
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def test_ok_status_for_good_video(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, status, error = process_video(video, "athlete_1", estimator)

        assert status == "ok"
        assert error is None
        assert row["athlete_id"] == "athlete_1"
        assert row["video_filename"] == "sample.mp4"

    def test_no_pose_status_for_empty_df(self, tmp_path):
        video = tmp_path / "empty.mp4"
        video.touch()
        estimator = MagicMock()
        estimator.process_video.return_value = pd.DataFrame()

        row, status, error = process_video(video, "athlete_1", estimator)

        assert status == "no_pose"
        assert error is None
        assert math.isnan(row["forty_time"])

    def test_error_status_on_exception(self, tmp_path):
        video = tmp_path / "bad.mp4"
        video.touch()
        estimator = MagicMock()
        estimator.process_video.side_effect = RuntimeError("corrupt video")

        row, status, error = process_video(video, "athlete_1", estimator)

        assert status == "error"
        assert "corrupt video" in error

    def test_row_has_all_output_columns(self, tmp_path):
        video = tmp_path / "full.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, status, _ = process_video(video, "ath", estimator)

        for col in _OUTPUT_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_data_quality_propagated(self, tmp_path):
        video = tmp_path / "vid.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="REVIEW"):
            row, _, _ = process_video(video, "ath", estimator)

        assert row["data_quality"] == "REVIEW"

    def test_forty_time_nan_when_out_of_range(self, tmp_path):
        """If analyze_velocity returns an implausible time it should be NaN in row."""
        video = tmp_path / "vid.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        out_of_range_vel = {
            "peak_velocity_mph": 20.0,
            "peak_velocity_ms": 9.0,
            "avg_velocity_ms": 8.0,
            "forty_time": 2.0,  # Below _TIME_MIN
        }
        with patch("athlete_vision.pipeline.analyze_velocity", return_value=out_of_range_vel), \
             patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, _, _ = process_video(video, "ath", estimator)

        assert math.isnan(row["forty_time"])


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _write_videos(self, directory: Path, names: list[str]) -> None:
        for name in names:
            (directory / name).touch()

    def _stub_process_video(self, video_path, athlete_id, estimator, **kwargs):
        """Minimal successful process_video stub."""
        return (
            {
                "athlete_id": athlete_id,
                "video_filename": video_path.name,
                "forty_time": 4.4,
                "stride_length": 1.2,
                "stride_frequency": 2.5,
                "ground_contact_ms": 120.0,
                "drive_phase_angle": 45.0,
                "hip_extension": 160.0,
                "arm_swing_symmetry": 85.0,
                "forward_lean_angle": 30.0,
                "transition_point_yards": 12.0,
                "peak_velocity_mph": 20.5,
                "data_quality": "OK",
            },
            "ok",
            None,
        )

    def test_raises_when_no_videos(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pipeline(tmp_path, tmp_path / "out.csv")

    def test_csv_written(self, tmp_path):
        self._write_videos(tmp_path, ["a.mp4", "b.mp4"])
        output = tmp_path / "results.csv"

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=self._stub_process_video):
            run_pipeline(tmp_path, output)

        assert output.exists()
        loaded = pd.read_csv(output)
        assert len(loaded) == 2

    def test_returns_dataframe_and_stats(self, tmp_path):
        self._write_videos(tmp_path, ["x.mp4"])
        output = tmp_path / "out.csv"

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=self._stub_process_video):
            df, stats = run_pipeline(tmp_path, output)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "processed" in stats
        assert "flagged" in stats
        assert "failed" in stats

    def test_column_order_matches_spec(self, tmp_path):
        self._write_videos(tmp_path, ["v.mp4"])
        output = tmp_path / "out.csv"

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=self._stub_process_video):
            df, _ = run_pipeline(tmp_path, output)

        assert list(df.columns) == _OUTPUT_COLUMNS

    def test_stats_count_correctly(self, tmp_path):
        self._write_videos(tmp_path, ["ok.mp4", "flagged.mp4", "err.mp4"])
        output = tmp_path / "out.csv"

        def stub(video_path, athlete_id, estimator, **kwargs):
            name = video_path.name
            if name == "ok.mp4":
                row = self._stub_process_video(video_path, athlete_id, estimator)[0]
                return row, "ok", None
            if name == "flagged.mp4":
                row = self._stub_process_video(video_path, athlete_id, estimator)[0]
                row["data_quality"] = "REVIEW"
                return row, "ok", None
            row = {col: float("nan") for col in _OUTPUT_COLUMNS}
            row["athlete_id"] = athlete_id
            row["video_filename"] = name
            row["data_quality"] = "REVIEW"
            return row, "error", "boom"

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            df, stats = run_pipeline(tmp_path, output)

        assert stats["processed"] == 2   # ok + flagged both "processed"
        assert stats["flagged"] == 1     # only the flagged one
        assert stats["failed"] == 1      # error video

    def test_athlete_id_fallback_to_stem(self, tmp_path):
        self._write_videos(tmp_path, ["myvideo.mp4"])
        output = tmp_path / "out.csv"

        captured_ids: list[str] = []

        def stub(video_path, athlete_id, estimator, **kwargs):
            captured_ids.append(athlete_id)
            return self._stub_process_video(video_path, athlete_id, estimator)

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            run_pipeline(tmp_path, output)

        assert captured_ids[0] == "myvideo"

    def test_athlete_id_explicit(self, tmp_path):
        self._write_videos(tmp_path, ["v.mp4"])
        output = tmp_path / "out.csv"

        captured_ids: list[str] = []

        def stub(video_path, athlete_id, estimator, **kwargs):
            captured_ids.append(athlete_id)
            return self._stub_process_video(video_path, athlete_id, estimator)

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            run_pipeline(tmp_path, output, athlete_id="BOLT")

        assert captured_ids[0] == "BOLT"

    def test_discovers_all_supported_extensions(self, tmp_path):
        self._write_videos(tmp_path, ["a.mp4", "b.mov", "c.avi", "d.mkv", "e.txt"])
        output = tmp_path / "out.csv"

        processed_names: list[str] = []

        def stub(video_path, athlete_id, estimator, **kwargs):
            processed_names.append(video_path.name)
            return self._stub_process_video(video_path, athlete_id, estimator)

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            run_pipeline(tmp_path, output)

        assert "e.txt" not in processed_names
        assert len(processed_names) == 4

    def test_continues_after_per_video_error(self, tmp_path):
        """A single failed video must not stop the rest of the pipeline."""
        self._write_videos(tmp_path, ["good.mp4", "bad.mp4"])
        output = tmp_path / "out.csv"

        def stub(video_path, athlete_id, estimator, **kwargs):
            if video_path.name == "bad.mp4":
                row = {col: float("nan") for col in _OUTPUT_COLUMNS}
                row["athlete_id"] = athlete_id
                row["video_filename"] = video_path.name
                row["data_quality"] = "REVIEW"
                return row, "error", "boom"
            return self._stub_process_video(video_path, athlete_id, estimator)

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            df, stats = run_pipeline(tmp_path, output)

        assert len(df) == 2
        assert stats["failed"] == 1
        assert stats["processed"] == 1


# ---------------------------------------------------------------------------
# print_pipeline_summary
# ---------------------------------------------------------------------------

class TestPrintPipelineSummary:
    def _make_df(self, n: int = 4, times=None, vels=None) -> pd.DataFrame:
        data = {col: [float("nan")] * n for col in _OUTPUT_COLUMNS}
        data["athlete_id"] = [f"ath{i}" for i in range(n)]
        data["video_filename"] = [f"v{i}.mp4" for i in range(n)]
        data["data_quality"] = ["OK"] * n
        if times is not None:
            data["forty_time"] = times
        if vels is not None:
            data["peak_velocity_mph"] = vels
        return pd.DataFrame(data)

    def test_runs_without_error(self, capsys):
        df = self._make_df()
        print_pipeline_summary(df, {"processed": 3, "flagged": 1, "failed": 1})
        out = capsys.readouterr().out
        assert "Pipeline Summary" in out

    def test_shows_counts(self, capsys):
        df = self._make_df()
        print_pipeline_summary(df, {"processed": 3, "flagged": 1, "failed": 2})
        out = capsys.readouterr().out
        assert "3" in out
        assert "1" in out
        assert "2" in out

    def test_shows_time_distribution_when_present(self, capsys):
        df = self._make_df(times=[4.2, 4.4, 4.6, 4.8])
        print_pipeline_summary(df, {"processed": 4, "flagged": 0, "failed": 0})
        out = capsys.readouterr().out
        assert "40-Yard" in out
        assert "Mean" in out

    def test_no_time_section_when_all_nan(self, capsys):
        df = self._make_df()  # all forty_time are NaN
        print_pipeline_summary(df, {"processed": 4, "flagged": 0, "failed": 0})
        out = capsys.readouterr().out
        assert "40-Yard" not in out

    def test_shows_velocity_section_when_present(self, capsys):
        df = self._make_df(vels=[18.0, 20.0, 22.0, 19.5])
        print_pipeline_summary(df, {"processed": 4, "flagged": 0, "failed": 0})
        out = capsys.readouterr().out
        assert "mph" in out

    def test_no_velocity_section_when_all_nan(self, capsys):
        df = self._make_df()  # all peak_velocity_mph are NaN
        print_pipeline_summary(df, {"processed": 4, "flagged": 0, "failed": 0})
        out = capsys.readouterr().out
        assert "mph" not in out
