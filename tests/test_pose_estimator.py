"""Tests for PoseEstimator."""

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from athlete_vision.pose_estimator import TRACKED_LANDMARKS, PoseEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_video(path: Path, n_frames: int = 5, width: int = 64, height: int = 48) -> None:
    """Write a minimal valid AVI so OpenCV can open it."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_landmark(x=0.5, y=0.5, z=0.0, visibility=0.9):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


def _make_pose_landmarks(n_landmarks: int = 33):
    landmarks = MagicMock()
    landmarks.landmark = [_make_landmark() for _ in range(n_landmarks)]
    return landmarks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackedLandmarks:
    def test_expected_joints_present(self):
        expected = {
            "left_ankle", "right_ankle",
            "left_knee", "right_knee",
            "left_hip", "right_hip",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        }
        assert set(TRACKED_LANDMARKS.keys()) == expected

    def test_indices_are_valid_mediapipe_indices(self):
        for joint, idx in TRACKED_LANDMARKS.items():
            assert 0 <= idx <= 32, f"{joint} index {idx} out of MediaPipe range"


class TestPoseEstimatorContextManager:
    def test_enter_exit(self):
        mock_instance = MagicMock()
        with patch("athlete_vision.pose_estimator.PoseLandmarker") as mock_landmarker_cls, \
             patch("athlete_vision.pose_estimator._find_model", return_value="/fake/model.task"):
            mock_landmarker_cls.create_from_options.return_value = mock_instance
            with PoseEstimator() as estimator:
                assert estimator is not None
        mock_instance.close.assert_called_once()


class TestProcessVideo:
    def _run_with_mock_pose(self, video_path, pose_result_factory, n_frames=3):
        """Run process_video with a mocked MediaPipe Pose that returns results
        produced by pose_result_factory(frame_index)."""
        results = [pose_result_factory(i) for i in range(n_frames)]
        call_count = {"n": 0}

        def fake_process(rgb):
            idx = call_count["n"]
            call_count["n"] += 1
            return results[idx] if idx < len(results) else MagicMock(pose_landmarks=None)

        with patch("mediapipe.solutions.pose.Pose") as mock_pose_cls:
            mock_instance = MagicMock()
            mock_instance.process.side_effect = fake_process
            mock_pose_cls.return_value = mock_instance

            estimator = PoseEstimator()
            df = estimator.process_video(str(video_path))
            estimator.close()
        return df

    def test_returns_dataframe(self, tmp_path):
        video = tmp_path / "test.avi"
        _make_fake_video(video, n_frames=3)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = _make_pose_landmarks()
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=3)
        assert isinstance(df, pd.DataFrame)

    def test_frame_count_matches(self, tmp_path):
        video = tmp_path / "test.avi"
        n = 5
        _make_fake_video(video, n_frames=n)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = _make_pose_landmarks()
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=n)
        assert len(df) == n

    def test_columns_present_when_landmarks_detected(self, tmp_path):
        video = tmp_path / "test.avi"
        _make_fake_video(video, n_frames=1)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = _make_pose_landmarks()
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=1)
        for joint in TRACKED_LANDMARKS:
            for axis in ("x", "y", "z", "visibility"):
                assert f"{joint}_{axis}" in df.columns, f"Missing column {joint}_{axis}"

    def test_nan_when_no_landmarks(self, tmp_path):
        video = tmp_path / "test.avi"
        _make_fake_video(video, n_frames=2)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = None
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=2)
        for joint in TRACKED_LANDMARKS:
            assert df[f"{joint}_x"].isna().all()

    def test_timestamp_increases(self, tmp_path):
        video = tmp_path / "test.avi"
        _make_fake_video(video, n_frames=3)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = None
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=3)
        assert list(df["timestamp_sec"]) == sorted(df["timestamp_sec"].tolist())
        assert df["timestamp_sec"].iloc[0] == 0.0

    def test_avg_confidence_in_attrs(self, tmp_path):
        video = tmp_path / "test.avi"
        _make_fake_video(video, n_frames=2)

        def factory(i):
            r = MagicMock()
            r.pose_landmarks = _make_pose_landmarks()
            return r

        df = self._run_with_mock_pose(video, factory, n_frames=2)
        assert "avg_confidence" in df.attrs
        assert set(df.attrs["avg_confidence"].keys()) == set(TRACKED_LANDMARKS.keys())

    def test_invalid_path_raises(self):
        with patch("mediapipe.solutions.pose.Pose"):
            estimator = PoseEstimator()
            with pytest.raises(ValueError, match="Cannot open video"):
                estimator.process_video("/nonexistent/path/video.mp4")
            estimator.close()
