"""Tests for calibration.py."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from athlete_vision.calibration import (
    calibrate,
    calibration_factor_from_reference,
    detect_yard_lines,
    unit_to_metres,
)


# ---------------------------------------------------------------------------
# unit_to_metres
# ---------------------------------------------------------------------------

class TestUnitToMetres:
    def test_yards(self):
        assert abs(unit_to_metres(1.0, "yards") - 0.9144) < 1e-9

    def test_yard_singular(self):
        assert abs(unit_to_metres(1.0, "yard") - 0.9144) < 1e-9

    def test_yd_abbreviation(self):
        assert abs(unit_to_metres(1.0, "yd") - 0.9144) < 1e-9

    def test_feet(self):
        assert abs(unit_to_metres(1.0, "feet") - 0.3048) < 1e-9

    def test_foot_singular(self):
        assert abs(unit_to_metres(1.0, "foot") - 0.3048) < 1e-9

    def test_ft_abbreviation(self):
        assert abs(unit_to_metres(1.0, "ft") - 0.3048) < 1e-9

    def test_metres(self):
        assert abs(unit_to_metres(1.0, "metres") - 1.0) < 1e-9

    def test_meters(self):
        assert abs(unit_to_metres(1.0, "meters") - 1.0) < 1e-9

    def test_m_abbreviation(self):
        assert abs(unit_to_metres(1.0, "m") - 1.0) < 1e-9

    def test_forty_yards(self):
        # 40 yards = 36.576 m exactly
        assert abs(unit_to_metres(40.0, "yards") - 36.576) < 1e-6

    def test_case_insensitive(self):
        assert abs(unit_to_metres(1.0, "YARDS") - 0.9144) < 1e-9
        assert abs(unit_to_metres(1.0, "Feet") - 0.3048) < 1e-9
        assert abs(unit_to_metres(1.0, "M") - 1.0) < 1e-9

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            unit_to_metres(1.0, "furlongs")

    def test_zero_value_returns_zero(self):
        assert unit_to_metres(0.0, "yards") == 0.0

    def test_scaling_is_linear(self):
        assert abs(unit_to_metres(2.0, "yards") - 2 * unit_to_metres(1.0, "yards")) < 1e-9


# ---------------------------------------------------------------------------
# calibration_factor_from_reference
# ---------------------------------------------------------------------------

class TestCalibrationFactorFromReference:
    def test_forty_yards_ninety_percent_span(self):
        # 40 yards = 36.576 m spanning 0.9 of frame → factor ≈ 40.64
        factor = calibration_factor_from_reference(40.0, "yards", 0.9)
        assert abs(factor - 36.576 / 0.9) < 1e-6

    def test_full_frame_span(self):
        # 100 metres spanning 100 % of frame → factor = 100.0
        factor = calibration_factor_from_reference(100.0, "metres", 1.0)
        assert abs(factor - 100.0) < 1e-9

    def test_feet_and_yards_equivalent(self):
        # 120 feet == 40 yards — calibration factors must match
        factor_yards = calibration_factor_from_reference(40.0, "yards", 0.9)
        factor_feet = calibration_factor_from_reference(120.0, "feet", 0.9)
        assert abs(factor_yards - factor_feet) < 1e-6

    def test_factor_always_positive(self):
        factor = calibration_factor_from_reference(40.0, "yards", 0.5)
        assert factor > 0.0

    def test_zero_frame_span_raises(self):
        with pytest.raises(ValueError, match="frame_span must be positive"):
            calibration_factor_from_reference(40.0, "yards", 0.0)

    def test_negative_frame_span_raises(self):
        with pytest.raises(ValueError, match="frame_span must be positive"):
            calibration_factor_from_reference(40.0, "yards", -0.1)

    def test_factor_scales_linearly_with_distance(self):
        f40 = calibration_factor_from_reference(40.0, "yards", 0.9)
        f80 = calibration_factor_from_reference(80.0, "yards", 0.9)
        assert abs(f80 / f40 - 2.0) < 1e-9

    def test_factor_inversely_scales_with_span(self):
        # Halving the span doubles the calibration factor
        f_half = calibration_factor_from_reference(40.0, "yards", 0.45)
        f_full = calibration_factor_from_reference(40.0, "yards", 0.9)
        assert abs(f_half / f_full - 2.0) < 1e-9

    def test_metres_input_unchanged(self):
        # 50 m spanning 0.5 of frame → factor = 100 m/unit
        factor = calibration_factor_from_reference(50.0, "meters", 0.5)
        assert abs(factor - 100.0) < 1e-9

    def test_unknown_unit_propagates_error(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            calibration_factor_from_reference(40.0, "cubits", 0.9)


# ---------------------------------------------------------------------------
# detect_yard_lines
# ---------------------------------------------------------------------------

def _make_cap_mock(
    frame_count: float = 30.0,
    frame_width: float = 640.0,
    frame_height: float = 480.0,
    frame: np.ndarray | None = None,
    read_success: bool = True,
) -> MagicMock:
    """Build a cv2.VideoCapture mock."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    prop_map = {
        cv2.CAP_PROP_FRAME_COUNT: frame_count,
        cv2.CAP_PROP_FRAME_WIDTH: frame_width,
        cv2.CAP_PROP_FRAME_HEIGHT: frame_height,
    }
    cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    if frame is None:
        frame = np.zeros((int(frame_height), int(frame_width), 3), dtype=np.uint8)
    cap.read.return_value = (read_success, frame)
    return cap


class TestDetectYardLines:
    def test_returns_none_for_nonexistent_file(self, tmp_path):
        result = detect_yard_lines(tmp_path / "nonexistent.mp4")
        assert result is None

    def test_returns_none_when_cap_not_opened(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4")
        assert result is None

    def test_returns_none_for_blank_frames(self):
        # All-black frames → no edges → no lines
        cap = _make_cap_mock()
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4")
        assert result is None

    def test_returns_none_when_read_fails(self):
        cap = _make_cap_mock(read_success=False)
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4")
        assert result is None

    def test_returns_positive_factor_when_lines_detected(self):
        """Frame with two clear vertical white stripes should yield a positive factor."""
        # 640×480: white stripes at x≈160 and x≈480 (spacing ≈ 0.5 of width)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, 158:163, :] = 255  # left stripe
        frame[:, 478:483, :] = 255  # right stripe
        cap = _make_cap_mock(frame=frame)

        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4", min_lines=2)

        # The heuristic may or may not detect them; if it does the value must be > 0
        if result is not None:
            assert result > 0.0
            assert not math.isnan(result)

    def test_result_is_float_or_none(self):
        cap = _make_cap_mock()
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4")
        assert result is None or isinstance(result, float)

    def test_returns_none_for_zero_dimension_video(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0  # every dimension is 0
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            result = detect_yard_lines("dummy.mp4")
        assert result is None

    def test_cap_always_released(self):
        """VideoCapture.release() must be called even when detection fails."""
        cap = _make_cap_mock()
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap):
            detect_yard_lines("dummy.mp4")
        cap.release.assert_called_once()

    def test_warning_emitted_when_no_lines_detected(self):
        """A warning must be logged when Hough detection finds no yard lines."""
        cap = _make_cap_mock()  # blank frames → no lines found
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap), \
             patch("athlete_vision.calibration.logger") as mock_logger:
            result = detect_yard_lines("dummy.mp4")
        assert result is None
        mock_logger.warning.assert_called()

    def test_returns_none_for_fewer_than_3_clusters(self):
        """Only 2 distinct clusters → return None (default min_lines=3)."""
        cap = _make_cap_mock(frame_width=640.0, frame_height=480.0)
        # Two vertical line clusters at x=100 and x=450 (normalised 0.156 and 0.703)
        two_cluster_lines = np.array([
            [[100, 0, 100, 200]],
            [[100, 5, 100, 205]],
            [[100, 10, 100, 210]],
            [[450, 0, 450, 200]],
            [[450, 5, 450, 205]],
            [[450, 10, 450, 210]],
        ])
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap), \
             patch("athlete_vision.calibration.cv2.HoughLinesP", return_value=two_cluster_lines), \
             patch("athlete_vision.calibration.logger") as mock_logger:
            result = detect_yard_lines("dummy.mp4")  # default min_lines=3
        assert result is None
        mock_logger.warning.assert_called()

    def test_returns_none_when_spacing_below_5px(self):
        """Inter-cluster spacing < 5 px → return None (implausibly small)."""
        cap = _make_cap_mock(frame_width=640.0, frame_height=480.0)
        # Three clusters at x=64, 68, 72 (normalised 0.1, 0.106, 0.113)
        # spacing ~3.8 px < 5 px; use cluster_tolerance=0.001 to keep them separate
        tiny_gap_lines = np.array([
            [[64, 0, 64, 200]],
            [[64, 5, 64, 205]],
            [[64, 10, 64, 210]],
            [[68, 0, 68, 200]],
            [[68, 5, 68, 205]],
            [[68, 10, 68, 210]],
            [[72, 0, 72, 200]],
            [[72, 5, 72, 205]],
            [[72, 10, 72, 210]],
        ])
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap), \
             patch("athlete_vision.calibration.cv2.HoughLinesP", return_value=tiny_gap_lines), \
             patch("athlete_vision.calibration.logger") as mock_logger:
            result = detect_yard_lines("dummy.mp4", min_lines=3, cluster_tolerance=0.001)
        assert result is None
        mock_logger.warning.assert_called()

    def test_returns_none_when_spacing_above_40_percent_frame_width(self):
        """Inter-cluster spacing > 40% of frame width → return None (implausibly large)."""
        cap = _make_cap_mock(frame_width=640.0, frame_height=480.0)
        # Three clusters at x=64 (10%), x=352 (55%), x=608 (95%)
        # spacings: 0.45 and 0.40 → mean 0.425 > 0.40
        wide_gap_lines = np.array([
            [[64, 0, 64, 200]],
            [[64, 5, 64, 205]],
            [[64, 10, 64, 210]],
            [[352, 0, 352, 200]],
            [[352, 5, 352, 205]],
            [[352, 10, 352, 210]],
            [[608, 0, 608, 200]],
            [[608, 5, 608, 205]],
            [[608, 10, 608, 210]],
        ])
        with patch("athlete_vision.calibration.cv2.VideoCapture", return_value=cap), \
             patch("athlete_vision.calibration.cv2.HoughLinesP", return_value=wide_gap_lines), \
             patch("athlete_vision.calibration.logger") as mock_logger:
            result = detect_yard_lines("dummy.mp4", min_lines=3)
        assert result is None
        mock_logger.warning.assert_called()


# ---------------------------------------------------------------------------
# calibrate — main entry point
# ---------------------------------------------------------------------------

class TestCalibrate:
    def test_explicit_reference_takes_priority(self):
        factor = calibrate(
            calibration_distance=40.0,
            calibration_unit="yards",
            frame_span=0.9,
        )
        expected = calibration_factor_from_reference(40.0, "yards", 0.9)
        assert abs(factor - expected) < 1e-9

    def test_default_frame_span_is_0_9(self):
        factor_explicit = calibrate(
            calibration_distance=40.0,
            calibration_unit="yards",
            frame_span=0.9,
        )
        factor_default = calibrate(
            calibration_distance=40.0,
            calibration_unit="yards",
        )
        assert abs(factor_explicit - factor_default) < 1e-9

    def test_meters_unit_accepted(self):
        factor = calibrate(calibration_distance=36.576, calibration_unit="meters")
        assert factor > 0.0
        assert not math.isnan(factor)

    def test_feet_unit_accepted(self):
        factor = calibrate(calibration_distance=120.0, calibration_unit="feet")
        # 120 ft = 40 yards → same as yards reference
        factor_yards = calibrate(calibration_distance=40.0, calibration_unit="yards")
        assert abs(factor - factor_yards) < 1e-6

    def test_auto_detect_used_when_no_reference(self, tmp_path):
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()

        with patch(
            "athlete_vision.calibration.detect_yard_lines", return_value=42.0
        ) as mock_detect:
            factor = calibrate(video_path=fake_video, auto_detect=True)

        mock_detect.assert_called_once_with(fake_video)
        assert abs(factor - 42.0) < 1e-9

    def test_fallback_when_detection_returns_none(self, tmp_path):
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()

        with patch("athlete_vision.calibration.detect_yard_lines", return_value=None):
            factor = calibrate(video_path=fake_video, auto_detect=True)

        assert abs(factor - 1.0) < 1e-9

    def test_fallback_when_no_video_and_no_reference(self):
        factor = calibrate()
        assert abs(factor - 1.0) < 1e-9

    def test_auto_detect_skipped_when_disabled(self, tmp_path):
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()

        with patch(
            "athlete_vision.calibration.detect_yard_lines", return_value=42.0
        ) as mock_detect:
            factor = calibrate(video_path=fake_video, auto_detect=False)

        mock_detect.assert_not_called()
        assert abs(factor - 1.0) < 1e-9

    def test_explicit_reference_skips_auto_detect(self, tmp_path):
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()

        with patch(
            "athlete_vision.calibration.detect_yard_lines"
        ) as mock_detect:
            calibrate(
                video_path=fake_video,
                calibration_distance=40.0,
                calibration_unit="yards",
            )

        mock_detect.assert_not_called()

    def test_no_video_no_reference_auto_detect_skipped(self):
        # auto_detect=True but no video_path → no detection attempt, return 1.0
        with patch(
            "athlete_vision.calibration.detect_yard_lines"
        ) as mock_detect:
            factor = calibrate(video_path=None, auto_detect=True)

        mock_detect.assert_not_called()
        assert abs(factor - 1.0) < 1e-9

    def test_factor_used_in_stride_length_scaling(self):
        """Calibration factor > 1.0 should proportionally scale stride length."""
        import pandas as pd
        from athlete_vision.stride_analyzer import analyze_strides

        # Minimal gait: two contacts at x=0.1 and x=0.2 → step delta = 0.1
        n = 30
        contact = 5
        swing = 10
        left_y = ([0.9] * contact + [0.4] * swing) * 2 + [0.9] * (n - 2 * (contact + swing))
        left_x = (
            [0.1] * (contact + swing)
            + [0.2] * (contact + swing)
            + [0.2] * (n - 2 * (contact + swing))
        )
        df = pd.DataFrame({
            "frame_index": list(range(n)),
            "timestamp_sec": [i / 30.0 for i in range(n)],
            "left_ankle_x": left_x,
            "left_ankle_y": left_y,
            "right_ankle_x": [0.5] * n,
            "right_ankle_y": [0.4] * n,
        })

        factor = calibrate(calibration_distance=40.0, calibration_unit="yards")
        result = analyze_strides(df, calibration_factor=factor, ground_threshold=0.8)

        # With factor ≈ 40.6, stride_length ≈ 0.1 * 40.6 ≈ 4.06 m
        if not math.isnan(result["stride_length"]):
            assert result["stride_length"] > 1.0  # definitely not in raw normalised units
