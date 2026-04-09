"""Tests for stride_analyzer."""

import math

import numpy as np
import pandas as pd
import pytest

from athlete_vision.stride_analyzer import (
    _detect_contact_phases,
    _strides_for_foot,
    analyze_strides,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_frames: int,
    left_y: list[float],
    right_y: list[float],
    left_x: list[float] | None = None,
    right_x: list[float] | None = None,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a minimal keypoint DataFrame with ankle columns and timestamps."""
    assert len(left_y) == n_frames
    assert len(right_y) == n_frames
    left_x = left_x or [0.5] * n_frames
    right_x = right_x or [0.5] * n_frames
    timestamps = [i / fps for i in range(n_frames)]
    return pd.DataFrame(
        {
            "frame_index": list(range(n_frames)),
            "timestamp_sec": timestamps,
            "left_ankle_x": left_x,
            "left_ankle_y": left_y,
            "right_ankle_x": right_x,
            "right_ankle_y": right_y,
        }
    )


def _gait_pattern(
    n_strides: int,
    fps: float = 30.0,
    contact_frames: int = 5,
    swing_frames: int = 10,
    x_step: float = 0.1,
) -> pd.DataFrame:
    """Synthesise a simple gait pattern for one foot (left only for simplicity).

    Each stride: <contact_frames> frames at y=0.9 (ground), then
    <swing_frames> frames at y=0.4 (air).  x advances by x_step each stride.
    """
    left_y: list[float] = []
    right_y: list[float] = []
    left_x: list[float] = []

    x = 0.1
    for _ in range(n_strides + 1):  # +1 so the last stride has a closing contact
        left_y.extend([0.9] * contact_frames)
        right_y.extend([0.5] * contact_frames)  # right foot always in air
        left_x.extend([x] * contact_frames)
        x += x_step
        left_y.extend([0.4] * swing_frames)
        right_y.extend([0.5] * swing_frames)
        left_x.extend([x] * swing_frames)

    n = len(left_y)
    return _make_df(n, left_y, right_y, left_x=left_x, fps=fps)


# ---------------------------------------------------------------------------
# _detect_contact_phases
# ---------------------------------------------------------------------------

class TestDetectContactPhases:
    def test_single_contact(self):
        y = pd.Series([0.3, 0.9, 0.9, 0.9, 0.3])
        phases = _detect_contact_phases(y, threshold=0.8)
        assert phases == [(1, 3)]

    def test_two_contacts(self):
        y = pd.Series([0.9, 0.9, 0.3, 0.3, 0.9, 0.9])
        phases = _detect_contact_phases(y, threshold=0.8)
        assert len(phases) == 2
        assert phases[0] == (0, 1)
        assert phases[1] == (4, 5)

    def test_contact_at_end(self):
        y = pd.Series([0.3, 0.3, 0.9, 0.9])
        phases = _detect_contact_phases(y, threshold=0.8)
        assert phases == [(2, 3)]

    def test_no_contact(self):
        y = pd.Series([0.3, 0.4, 0.3])
        phases = _detect_contact_phases(y, threshold=0.8)
        assert phases == []

    def test_entire_series_in_contact(self):
        y = pd.Series([0.9, 0.9, 0.9])
        phases = _detect_contact_phases(y, threshold=0.8)
        assert phases == [(0, 2)]


# ---------------------------------------------------------------------------
# analyze_strides – edge cases
# ---------------------------------------------------------------------------

class TestAnalyzeStridesEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = analyze_strides(df)
        assert result["strides"] == []
        assert math.isnan(result["stride_length"])
        assert math.isnan(result["stride_frequency"])
        assert math.isnan(result["ground_contact_ms"])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp_sec": [0.0], "left_ankle_x": [0.5]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_strides(df)

    def test_no_strides_detected(self):
        # Ankle y never reaches ground → no contact phases
        df = _make_df(10, [0.3] * 10, [0.3] * 10)
        result = analyze_strides(df, ground_threshold=0.8)
        assert result["strides"] == []
        assert math.isnan(result["stride_length"])

    def test_single_contact_no_stride(self):
        # Only one ground contact phase → can't form a complete stride cycle
        y = [0.9] * 5 + [0.3] * 20
        df = _make_df(25, y, [0.3] * 25)
        result = analyze_strides(df, ground_threshold=0.8)
        assert result["strides"] == []


# ---------------------------------------------------------------------------
# analyze_strides – metric correctness
# ---------------------------------------------------------------------------

class TestAnalyzeStridesMetrics:
    def test_stride_count(self):
        n_strides = 3
        df = _gait_pattern(n_strides, fps=30.0, contact_frames=5, swing_frames=10)
        result = analyze_strides(df, ground_threshold=0.8)
        # n_strides strides should be detected for the left foot
        left_strides = [s for s in result["strides"] if s["foot"] == "left"]
        assert len(left_strides) == n_strides

    def test_stride_length_with_calibration(self):
        x_step = 0.2
        df = _gait_pattern(
            n_strides=2,
            fps=30.0,
            contact_frames=5,
            swing_frames=10,
            x_step=x_step,
        )
        # calibration_factor = 10.0 means 1 normalised unit = 10 metres
        result = analyze_strides(df, calibration_factor=10.0, ground_threshold=0.8)
        left_strides = [s for s in result["strides"] if s["foot"] == "left"]
        for s in left_strides:
            assert abs(s["stride_length_m"] - x_step * 10.0) < 1e-6

    def test_stride_frequency(self):
        fps = 30.0
        contact_frames = 5
        swing_frames = 10
        stride_frames = contact_frames + swing_frames
        expected_freq = fps / stride_frames  # strides per second

        df = _gait_pattern(
            n_strides=3,
            fps=fps,
            contact_frames=contact_frames,
            swing_frames=swing_frames,
        )
        result = analyze_strides(df, ground_threshold=0.8)
        left_strides = [s for s in result["strides"] if s["foot"] == "left"]
        for s in left_strides:
            assert abs(s["stride_frequency_hz"] - expected_freq) < 1e-3

    def test_ground_contact_time(self):
        fps = 30.0
        contact_frames = 6
        swing_frames = 12
        expected_contact_ms = (contact_frames - 1) / fps * 1000.0

        df = _gait_pattern(
            n_strides=2,
            fps=fps,
            contact_frames=contact_frames,
            swing_frames=swing_frames,
        )
        result = analyze_strides(df, ground_threshold=0.8)
        left_strides = [s for s in result["strides"] if s["foot"] == "left"]
        for s in left_strides:
            assert abs(s["ground_contact_ms"] - expected_contact_ms) < 1.0

    def test_averages_match_per_stride(self):
        df = _gait_pattern(n_strides=4, fps=30.0)
        result = analyze_strides(df, ground_threshold=0.8)

        strides = result["strides"]
        assert abs(
            result["stride_length"]
            - np.mean([s["stride_length_m"] for s in strides])
        ) < 1e-9
        assert abs(
            result["stride_frequency"]
            - np.mean([s["stride_frequency_hz"] for s in strides])
        ) < 1e-9
        assert abs(
            result["ground_contact_ms"]
            - np.mean([s["ground_contact_ms"] for s in strides])
        ) < 1e-9

    def test_adaptive_threshold(self):
        # Without explicit threshold the function should still detect strides
        df = _gait_pattern(n_strides=2, fps=30.0)
        result = analyze_strides(df)  # no ground_threshold
        assert len(result["strides"]) > 0

    def test_per_stride_dict_keys(self):
        df = _gait_pattern(n_strides=1, fps=30.0)
        result = analyze_strides(df, ground_threshold=0.8)
        expected_keys = {
            "foot",
            "stride_index",
            "start_frame",
            "end_frame",
            "stride_length_m",
            "stride_frequency_hz",
            "ground_contact_ms",
        }
        for stride in result["strides"]:
            assert expected_keys == set(stride.keys())

    def test_both_feet_detected(self):
        # Build a DataFrame with both feet alternating contact
        fps = 30.0
        contact = 5
        swing = 10
        n = (contact + swing) * 4
        left_y = []
        right_y = []
        left_x = []
        right_x = []
        x_l, x_r = 0.1, 0.2
        step = 0.1
        for i in range(4):
            if i % 2 == 0:
                left_y += [0.9] * contact + [0.4] * swing
                right_y += [0.4] * contact + [0.9] * swing
            else:
                left_y += [0.4] * contact + [0.9] * swing
                right_y += [0.9] * contact + [0.4] * swing
            left_x += [x_l] * (contact + swing)
            right_x += [x_r] * (contact + swing)
            x_l += step
            x_r += step

        df = _make_df(n, left_y, right_y, left_x=left_x, right_x=right_x, fps=fps)
        result = analyze_strides(df, ground_threshold=0.8)
        feet = {s["foot"] for s in result["strides"]}
        assert "left" in feet
        assert "right" in feet
