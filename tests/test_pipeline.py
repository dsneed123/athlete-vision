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
    filter_low_confidence_frames,
    print_pipeline_summary,
    process_video,
    run_pipeline,
    validate_pose_plausibility,
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
# filter_low_confidence_frames
# ---------------------------------------------------------------------------

class TestFilterLowConfidenceFrames:
    """Unit tests for filter_low_confidence_frames()."""

    def _make_df(self, visibility: float, n_frames: int = 5) -> pd.DataFrame:
        """Minimal DataFrame with a single joint and configurable visibility."""
        data = {
            "left_ankle_x": [0.5] * n_frames,
            "left_ankle_y": [0.8] * n_frames,
            "left_ankle_z": [0.0] * n_frames,
            "left_ankle_visibility": [visibility] * n_frames,
        }
        return pd.DataFrame(data)

    def test_nan_filled_when_visibility_below_threshold(self):
        """x/y/z must be NaN for every frame whose visibility is below the threshold."""
        df = self._make_df(visibility=0.70)
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert result["left_ankle_x"].isna().all()
        assert result["left_ankle_y"].isna().all()
        assert result["left_ankle_z"].isna().all()

    def test_coordinates_intact_when_visibility_at_threshold(self):
        """Visibility exactly at the threshold must NOT be NaN-filled."""
        df = self._make_df(visibility=0.85)
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert not result["left_ankle_x"].isna().any()
        assert not result["left_ankle_y"].isna().any()
        assert not result["left_ankle_z"].isna().any()

    def test_coordinates_intact_when_visibility_above_threshold(self):
        """Visibility above the threshold must leave x/y/z unchanged."""
        df = self._make_df(visibility=0.95)
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert not result["left_ankle_x"].isna().any()
        assert not result["left_ankle_y"].isna().any()
        assert not result["left_ankle_z"].isna().any()

    def test_only_low_confidence_frames_are_nan_filled(self):
        """Only the frames below the threshold are affected; others stay intact."""
        n_frames = 10
        visibility = [0.90] * n_frames
        visibility[2] = 0.50   # frame 2 is below threshold
        visibility[7] = 0.30   # frame 7 is below threshold
        df = pd.DataFrame({
            "left_ankle_x": [0.5] * n_frames,
            "left_ankle_y": [0.8] * n_frames,
            "left_ankle_z": [0.0] * n_frames,
            "left_ankle_visibility": visibility,
        })
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        for i in range(n_frames):
            if i in (2, 7):
                assert math.isnan(result["left_ankle_x"].iloc[i])
                assert math.isnan(result["left_ankle_y"].iloc[i])
                assert math.isnan(result["left_ankle_z"].iloc[i])
            else:
                assert not math.isnan(result["left_ankle_x"].iloc[i])

    def test_missing_visibility_column_skipped_gracefully(self):
        """A joint with no visibility column must not cause an error."""
        df = pd.DataFrame({
            "left_ankle_x": [0.5],
            "left_ankle_y": [0.8],
            "left_ankle_z": [0.0],
            # no left_ankle_visibility column
        })
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert not result["left_ankle_x"].isna().any()

    def test_default_joints_cover_critical_joints(self):
        """When joints=None the six critical joints (hips/knees/ankles) are filtered."""
        df = _make_pose_df(n_frames=5, visibility=0.50)
        result = filter_low_confidence_frames(df, threshold=0.85)
        for joint in ("left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle"):
            assert result[f"{joint}_x"].isna().all(), f"{joint}_x should be NaN"
            assert result[f"{joint}_y"].isna().all(), f"{joint}_y should be NaN"

    def test_non_critical_joints_not_affected_by_default(self):
        """Joints outside the default list must not be NaN-filled."""
        df = _make_pose_df(n_frames=5, visibility=0.50)
        result = filter_low_confidence_frames(df, threshold=0.85)
        # Shoulders are not in the default critical joint list
        assert not result["left_shoulder_x"].isna().any()

    def test_original_dataframe_not_mutated(self):
        """The function must return a copy; the input must be unchanged."""
        df = self._make_df(visibility=0.50)
        original_values = df["left_ankle_x"].tolist()
        filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert df["left_ankle_x"].tolist() == original_values

    def test_attrs_preserved(self):
        """DataFrame attrs must be carried over to the returned copy."""
        df = self._make_df(visibility=0.95)
        df.attrs["avg_confidence"] = {"left_ankle": 0.95}
        result = filter_low_confidence_frames(df, threshold=0.85, joints=["left_ankle"])
        assert result.attrs["avg_confidence"] == {"left_ankle": 0.95}


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
# validate_pose_plausibility
# ---------------------------------------------------------------------------

class TestValidatePosePlausibility:
    """Tests for validate_pose_plausibility() covering each biomechanical check."""

    def test_plausible_with_valid_data(self):
        df = _make_pose_df(n_frames=120)
        assert validate_pose_plausibility(df) is True

    def test_empty_dataframe_is_plausible(self):
        assert validate_pose_plausibility(pd.DataFrame()) is True

    # --- Check 1: ankle below hip ---

    def test_ankle_above_hip_flagged(self):
        """Ankle y < hip y for all frames → implausible."""
        df = _make_pose_df(n_frames=120)
        # hip_y = 0.45; setting ankle_y = 0.30 puts the ankle *above* the hip
        df["left_ankle_y"] = 0.30
        assert validate_pose_plausibility(df) is False

    def test_ankle_above_hip_right_side_flagged(self):
        df = _make_pose_df(n_frames=120)
        df["right_ankle_y"] = 0.30
        assert validate_pose_plausibility(df) is False

    def test_ankle_above_hip_below_threshold_passes(self):
        """≤ 5 % of frames with ankle above hip should still pass."""
        df = _make_pose_df(n_frames=120)
        # 5 frames / 120 = 4.17 % < 5 % → should pass
        df.loc[:4, "left_ankle_y"] = 0.30
        assert validate_pose_plausibility(df) is True

    def test_ankle_above_hip_above_threshold_fails(self):
        """Just above 5 % (7/120 ≈ 5.83 %) should fail."""
        df = _make_pose_df(n_frames=120)
        df.loc[:6, "left_ankle_y"] = 0.30
        assert validate_pose_plausibility(df) is False

    # --- Check 2: knee between hip and ankle ---

    def test_knee_above_hip_flagged(self):
        """knee_y < hip_y means knee appears above hip → implausible."""
        df = _make_pose_df(n_frames=120)
        # hip_y = 0.45; knee_y = 0.30 puts knee above hip
        df["left_knee_y"] = 0.30
        assert validate_pose_plausibility(df) is False

    def test_knee_below_ankle_flagged(self):
        """knee_y > ankle_y means knee appears below ankle → implausible."""
        df = _make_pose_df(n_frames=120)
        # ankle_y oscillates in [0.7, 0.9]; 0.95 is always below
        df["left_knee_y"] = 0.95
        assert validate_pose_plausibility(df) is False

    def test_knee_right_side_outside_chain_flagged(self):
        df = _make_pose_df(n_frames=120)
        df["right_knee_y"] = 0.95
        assert validate_pose_plausibility(df) is False

    # --- Check 3: elbow between shoulder and wrist ---

    def test_elbow_below_shoulder_flagged(self):
        """elbow_y > max(shoulder_y, wrist_y) → elbow below both → implausible.

        In the synthetic data shoulder_y=0.30, wrist_y=0.10; setting
        elbow_y=0.50 puts it below the shoulder (outside the valid range).
        """
        df = _make_pose_df(n_frames=120)
        df["left_elbow_y"] = 0.50
        assert validate_pose_plausibility(df) is False

    def test_elbow_above_wrist_flagged(self):
        """elbow_y < min(shoulder_y, wrist_y) → elbow above both → implausible."""
        df = _make_pose_df(n_frames=120)
        # wrist_y=0.10, shoulder_y=0.30; setting elbow_y=0.05 puts it above both
        df["left_elbow_y"] = 0.05
        assert validate_pose_plausibility(df) is False

    def test_elbow_right_side_outside_range_flagged(self):
        df = _make_pose_df(n_frames=120)
        df["right_elbow_y"] = 0.50
        assert validate_pose_plausibility(df) is False

    # --- Check 4: left joints not to the right of right joints ---

    def test_mirrored_joints_flagged(self):
        """All left x > right x for every frame → mirrored pose → implausible."""
        df = _make_pose_df(n_frames=120)
        for joint in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist"):
            df[f"left_{joint}_x"] = 0.7
            df[f"right_{joint}_x"] = 0.3
        assert validate_pose_plausibility(df) is False

    def test_equal_left_right_x_passes(self):
        """Exactly equal left and right x (typical side-view) is not implausible."""
        df = _make_pose_df(n_frames=120)
        # _make_pose_df already sets left_x == right_x for every joint
        assert validate_pose_plausibility(df) is True

    # --- NaN tolerance ---

    def test_all_nan_joints_treated_as_plausible(self):
        """When all relevant joints are NaN the check is skipped (pass)."""
        df = _make_pose_df(n_frames=120)
        for col in df.columns:
            if col.endswith("_y") or col.endswith("_x"):
                df[col] = float("nan")
        assert validate_pose_plausibility(df) is True


# ---------------------------------------------------------------------------
# process_video — IMPLAUSIBLE_POSE integration
# ---------------------------------------------------------------------------

class TestProcessVideoImplausiblePose:
    """Integration tests ensuring validate_pose_plausibility is wired into process_video."""

    def test_implausible_pose_appended_to_ok_quality(self, tmp_path):
        """IMPLAUSIBLE_POSE is appended even when other quality checks pass."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        df["left_ankle_y"] = 0.30  # ankle above hip → implausible
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, status, error = process_video(video, "athlete_1", estimator)

        assert status == "ok"
        assert error is None
        assert row["data_quality"] == "OK|IMPLAUSIBLE_POSE"

    def test_implausible_pose_appended_to_review_quality(self, tmp_path):
        """IMPLAUSIBLE_POSE is appended to REVIEW when both conditions hold."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        df["left_ankle_y"] = 0.30
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="REVIEW"):
            row, _, _ = process_video(video, "athlete_1", estimator)

        assert row["data_quality"] == "REVIEW|IMPLAUSIBLE_POSE"

    def test_plausible_pose_leaves_quality_unchanged(self, tmp_path):
        """A plausible pose must not add IMPLAUSIBLE_POSE to data_quality."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, _, _ = process_video(video, "athlete_1", estimator)

        assert row["data_quality"] == "OK"
        assert "IMPLAUSIBLE_POSE" not in row["data_quality"]


# ---------------------------------------------------------------------------
# process_video — CROSS_BODY_ARM_SWING integration
# ---------------------------------------------------------------------------

class TestProcessVideoCrossBodyArmSwing:
    """Integration tests ensuring cross_body_swing is wired into data_quality."""

    def test_cross_body_flag_appended_when_detected(self, tmp_path):
        """CROSS_BODY_ARM_SWING is appended when arm_metrics reports it."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        arm_result = {
            "arm_swing_symmetry": 70.0,
            "left_arm_amplitude": 0.1,
            "right_arm_amplitude": 0.15,
            "left_cross_body_ratio": 0.2,
            "right_cross_body_ratio": 0.05,
            "cross_body_swing": True,
        }
        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"), \
             patch("athlete_vision.pipeline.analyze_arm_swing", return_value=arm_result):
            row, status, error = process_video(video, "athlete_1", estimator)

        assert status == "ok"
        assert "CROSS_BODY_ARM_SWING" in row["data_quality"]

    def test_cross_body_flag_not_appended_when_absent(self, tmp_path):
        """CROSS_BODY_ARM_SWING must not appear when cross_body_swing is False."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        arm_result = {
            "arm_swing_symmetry": 90.0,
            "left_arm_amplitude": 0.12,
            "right_arm_amplitude": 0.12,
            "left_cross_body_ratio": 0.02,
            "right_cross_body_ratio": 0.02,
            "cross_body_swing": False,
        }
        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"), \
             patch("athlete_vision.pipeline.analyze_arm_swing", return_value=arm_result):
            row, _, _ = process_video(video, "athlete_1", estimator)

        assert "CROSS_BODY_ARM_SWING" not in row["data_quality"]

    def test_cross_body_flag_combines_with_implausible_pose(self, tmp_path):
        """Both IMPLAUSIBLE_POSE and CROSS_BODY_ARM_SWING can appear together."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        df["left_ankle_y"] = 0.30  # ankle above hip → implausible
        estimator = MagicMock()
        estimator.process_video.return_value = df

        arm_result = {
            "arm_swing_symmetry": 60.0,
            "left_arm_amplitude": 0.1,
            "right_arm_amplitude": 0.2,
            "left_cross_body_ratio": 0.3,
            "right_cross_body_ratio": 0.05,
            "cross_body_swing": True,
        }
        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"), \
             patch("athlete_vision.pipeline.analyze_arm_swing", return_value=arm_result):
            row, _, _ = process_video(video, "athlete_1", estimator)

        assert "IMPLAUSIBLE_POSE" in row["data_quality"]
        assert "CROSS_BODY_ARM_SWING" in row["data_quality"]

    def test_cross_body_flag_with_review_quality(self, tmp_path):
        """CROSS_BODY_ARM_SWING appended to REVIEW base quality."""
        video = tmp_path / "sample.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        arm_result = {
            "arm_swing_symmetry": 65.0,
            "left_arm_amplitude": 0.1,
            "right_arm_amplitude": 0.18,
            "left_cross_body_ratio": 0.15,
            "right_cross_body_ratio": 0.05,
            "cross_body_swing": True,
        }
        with patch("athlete_vision.pipeline._check_data_quality", return_value="REVIEW"), \
             patch("athlete_vision.pipeline.analyze_arm_swing", return_value=arm_result):
            row, _, _ = process_video(video, "athlete_1", estimator)

        assert row["data_quality"] == "REVIEW|CROSS_BODY_ARM_SWING"


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

    def test_confidence_columns_present_and_in_range(self, tmp_path):
        """avg_confidence_hips/ankles/knees must be present and in [0.0, 1.0]."""
        video = tmp_path / "conf.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120, visibility=0.95)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, status, _ = process_video(video, "ath", estimator)

        assert status == "ok"
        for col in ("avg_confidence_hips", "avg_confidence_ankles", "avg_confidence_knees"):
            assert col in row, f"Missing column: {col}"
            assert 0.0 <= row[col] <= 1.0, f"{col}={row[col]} not in [0.0, 1.0]"

    def test_confidence_columns_nan_when_no_attrs(self, tmp_path):
        """When avg_confidence attrs are absent the columns must be NaN."""
        video = tmp_path / "noconf.mp4"
        video.touch()
        df = _make_pose_df(n_frames=120)
        df.attrs.pop("avg_confidence", None)
        estimator = MagicMock()
        estimator.process_video.return_value = df

        with patch("athlete_vision.pipeline._check_data_quality", return_value="OK"):
            row, status, _ = process_video(video, "ath", estimator)

        assert status == "ok"
        for col in ("avg_confidence_hips", "avg_confidence_ankles", "avg_confidence_knees"):
            assert math.isnan(row[col]), f"{col} should be NaN when attrs absent"


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
                "avg_confidence_hips": 0.95,
                "avg_confidence_ankles": 0.95,
                "avg_confidence_knees": 0.95,
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

    def test_stats_flagged_counts_review_with_implausible_pose(self, tmp_path):
        """REVIEW|IMPLAUSIBLE_POSE must count as flagged in stats."""
        self._write_videos(tmp_path, ["a.mp4", "b.mp4"])
        output = tmp_path / "out.csv"

        def stub(video_path, athlete_id, estimator, **kwargs):
            row = self._stub_process_video(video_path, athlete_id, estimator)[0]
            row["data_quality"] = "REVIEW|IMPLAUSIBLE_POSE"
            return row, "ok", None

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            _, stats = run_pipeline(tmp_path, output)

        assert stats["flagged"] == 2

    def test_stats_ok_with_implausible_pose_not_counted_as_review(self, tmp_path):
        """OK|IMPLAUSIBLE_POSE must not be counted as a REVIEW-flagged row."""
        self._write_videos(tmp_path, ["a.mp4"])
        output = tmp_path / "out.csv"

        def stub(video_path, athlete_id, estimator, **kwargs):
            row = self._stub_process_video(video_path, athlete_id, estimator)[0]
            row["data_quality"] = "OK|IMPLAUSIBLE_POSE"
            return row, "ok", None

        with patch("athlete_vision.pipeline.PoseEstimator", return_value=_make_pose_estimator_stub(pd.DataFrame())), \
             patch("athlete_vision.pipeline.process_video", side_effect=stub):
            _, stats = run_pipeline(tmp_path, output)

        assert stats["flagged"] == 0

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

    def test_shows_implausible_pose_count_when_present(self, capsys):
        data = {col: [float("nan")] * 3 for col in _OUTPUT_COLUMNS}
        data["athlete_id"] = ["a0", "a1", "a2"]
        data["video_filename"] = ["v0.mp4", "v1.mp4", "v2.mp4"]
        data["data_quality"] = ["OK", "OK|IMPLAUSIBLE_POSE", "REVIEW|IMPLAUSIBLE_POSE"]
        df = pd.DataFrame(data)
        print_pipeline_summary(df, {"processed": 3, "flagged": 1, "failed": 0})
        out = capsys.readouterr().out
        assert "Implausible pose" in out
        assert "2" in out

    def test_no_implausible_pose_line_when_none(self, capsys):
        df = self._make_df()  # all data_quality are "OK"
        print_pipeline_summary(df, {"processed": 4, "flagged": 0, "failed": 0})
        out = capsys.readouterr().out
        assert "Implausible pose" not in out
