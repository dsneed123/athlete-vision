"""Tests for velocity_analyzer."""

import math

import numpy as np
import pandas as pd
import pytest

from athlete_vision.velocity_analyzer import (
    _smooth_velocities,
    _detect_movement_window,
    analyze_velocity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_frames: int,
    hip_x: list[float],
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a minimal DataFrame with bilateral hip x and timestamps."""
    return pd.DataFrame(
        {
            "frame_index": list(range(n_frames)),
            "timestamp_sec": [i / fps for i in range(n_frames)],
            "left_hip_x": hip_x,
            "right_hip_x": [x + 0.05 for x in hip_x],  # right hip slightly offset
        }
    )


def _constant_velocity_df(
    velocity_norm: float,
    n_frames: int = 120,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Synthesise an athlete moving at constant normalised velocity."""
    dt = 1.0 / fps
    hip_x = [0.05 + velocity_norm * dt * i for i in range(n_frames)]
    return _make_df(n_frames, hip_x, fps=fps)


def _forty_yard_df(
    elapsed: float,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Synthesise a 40-yard run lasting *elapsed* seconds at constant pace."""
    n_frames = int(elapsed * fps) + 1
    # Uniform displacement per frame, calibration_factor will convert units
    hip_x = [i / (n_frames - 1) * 0.9 for i in range(n_frames)]
    return _make_df(n_frames, hip_x, fps=fps)


# ---------------------------------------------------------------------------
# _smooth_velocities
# ---------------------------------------------------------------------------

class TestSmoothVelocities:
    def test_constant_signal_unchanged(self):
        v = np.ones(20) * 5.0
        smoothed = _smooth_velocities(v, window=5)
        # Edges may differ; interior should be 5.0
        assert abs(float(np.nanmean(smoothed[2:-2])) - 5.0) < 1e-9

    def test_output_length_matches_input(self):
        v = np.arange(15, dtype=float)
        smoothed = _smooth_velocities(v, window=5)
        assert len(smoothed) == len(v)

    def test_short_signal_returned_unchanged(self):
        v = np.array([1.0, 2.0, 3.0])
        smoothed = _smooth_velocities(v, window=5)
        np.testing.assert_array_equal(smoothed, v)

    def test_smoothing_reduces_spikes(self):
        v = np.ones(20)
        v[10] = 100.0  # spike
        smoothed = _smooth_velocities(v, window=5)
        assert smoothed[10] < 100.0


# ---------------------------------------------------------------------------
# _detect_movement_window
# ---------------------------------------------------------------------------

class TestDetectMovementWindow:
    def test_constant_motion_returns_window(self):
        hip_x = pd.Series([0.1 * i for i in range(30)])
        result = _detect_movement_window(hip_x)
        assert result is not None
        start, end = result
        assert start < end

    def test_stationary_returns_none(self):
        hip_x = pd.Series([0.5] * 20)
        result = _detect_movement_window(hip_x)
        assert result is None

    def test_single_value_returns_none(self):
        hip_x = pd.Series([0.5])
        result = _detect_movement_window(hip_x)
        assert result is None


# ---------------------------------------------------------------------------
# analyze_velocity – edge cases
# ---------------------------------------------------------------------------

class TestAnalyzeVelocityEdgeCases:
    def test_empty_dataframe(self):
        result = analyze_velocity(pd.DataFrame())
        assert math.isnan(result["peak_velocity_mph"])
        assert math.isnan(result["peak_velocity_ms"])
        assert math.isnan(result["avg_velocity_ms"])
        assert math.isnan(result["forty_time"])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"timestamp_sec": [0.0, 1.0], "left_hip_x": [0.1, 0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_velocity(df)

    def test_single_frame_returns_nan(self):
        df = _make_df(1, [0.5])
        result = analyze_velocity(df)
        assert math.isnan(result["peak_velocity_mph"])

    def test_result_keys(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        result = analyze_velocity(df)
        expected_keys = {
            "peak_velocity_mph",
            "peak_velocity_ms",
            "avg_velocity_ms",
            "forty_time",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# analyze_velocity – metric correctness
# ---------------------------------------------------------------------------

class TestAnalyzeVelocityMetrics:
    def test_peak_velocity_positive(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        result = analyze_velocity(df, calibration_factor=10.0)
        assert result["peak_velocity_ms"] > 0.0
        assert result["peak_velocity_mph"] > 0.0

    def test_mph_conversion(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        result = analyze_velocity(df, calibration_factor=10.0)
        # 1 m/s = 2.23694 mph
        expected_mph = result["peak_velocity_ms"] * 2.23694
        assert abs(result["peak_velocity_mph"] - expected_mph) < 1e-4

    def test_peak_velocity_scales_with_calibration(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        r1 = analyze_velocity(df, calibration_factor=1.0)
        r2 = analyze_velocity(df, calibration_factor=2.0)
        assert abs(r2["peak_velocity_ms"] / r1["peak_velocity_ms"] - 2.0) < 0.05

    def test_faster_athlete_higher_velocity(self):
        slow_df = _constant_velocity_df(velocity_norm=0.005, n_frames=120)
        fast_df = _constant_velocity_df(velocity_norm=0.020, n_frames=120)
        slow_result = analyze_velocity(slow_df, calibration_factor=40.0)
        fast_result = analyze_velocity(fast_df, calibration_factor=40.0)
        assert fast_result["peak_velocity_ms"] > slow_result["peak_velocity_ms"]

    def test_avg_velocity_not_nan_when_moving(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        result = analyze_velocity(df, calibration_factor=10.0)
        assert not math.isnan(result["avg_velocity_ms"])
        assert result["avg_velocity_ms"] > 0.0

    def test_forty_time_within_plausible_range(self):
        # Build a run whose elapsed time is exactly 4.5 s
        elapsed = 4.5
        df = _forty_yard_df(elapsed=elapsed, fps=30.0)
        result = analyze_velocity(df, calibration_factor=36.576)
        # Time should be detected (within 3.5–6.5 s window)
        assert not math.isnan(result["forty_time"])
        assert 3.5 <= result["forty_time"] <= 6.5

    def test_forty_time_too_short_returns_nan(self):
        # 1-second clip is outside the 3.5–6.5 s plausible range
        df = _forty_yard_df(elapsed=1.0, fps=30.0)
        result = analyze_velocity(df)
        assert math.isnan(result["forty_time"])

    def test_stationary_athlete_near_zero_velocity(self):
        n = 60
        df = _make_df(n, [0.5] * n)
        result = analyze_velocity(df, calibration_factor=10.0)
        # May return 0 or nan — either is acceptable; should not be large
        if not math.isnan(result["peak_velocity_ms"]):
            assert result["peak_velocity_ms"] < 0.1

    def test_smooth_window_parameter(self):
        df = _constant_velocity_df(velocity_norm=0.01)
        r1 = analyze_velocity(df, smooth_window=1)
        r5 = analyze_velocity(df, smooth_window=5)
        # Both should return finite, positive values for a constant-velocity run
        assert not math.isnan(r1["peak_velocity_ms"])
        assert not math.isnan(r5["peak_velocity_ms"])
