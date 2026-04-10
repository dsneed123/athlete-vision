"""Tests for stride_analyzer."""

import math

import numpy as np
import pandas as pd
import pytest

from athlete_vision.stride_analyzer import (
    _detect_contact_phases,
    _has_long_nan_run,
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

    def test_normal_result_includes_gap_flag(self):
        # has_long_tracking_gap is present and False when data is clean
        df = _gait_pattern(n_strides=2, fps=30.0)
        result = analyze_strides(df, ground_threshold=0.8)
        assert result["has_long_tracking_gap"] is False


# ---------------------------------------------------------------------------
# _has_long_nan_run
# ---------------------------------------------------------------------------

class TestHasLongNanRun:
    def test_no_nans(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _has_long_nan_run(s, max_gap=5) is False

    def test_short_run_below_threshold(self):
        s = pd.Series([1.0, float("nan"), float("nan"), 1.0])
        assert _has_long_nan_run(s, max_gap=3) is False

    def test_run_exactly_at_threshold_not_triggered(self):
        # A run of exactly max_gap frames is NOT over the threshold
        s = pd.Series([1.0] + [float("nan")] * 5 + [1.0])
        assert _has_long_nan_run(s, max_gap=5) is False

    def test_run_one_over_threshold(self):
        s = pd.Series([1.0] + [float("nan")] * 6 + [1.0])
        assert _has_long_nan_run(s, max_gap=5) is True

    def test_multiple_short_runs_not_triggered(self):
        s = pd.Series([float("nan"), float("nan"), 1.0, float("nan"), float("nan")])
        assert _has_long_nan_run(s, max_gap=3) is False

    def test_run_at_end(self):
        s = pd.Series([1.0, 1.0] + [float("nan")] * 10)
        assert _has_long_nan_run(s, max_gap=5) is True


# ---------------------------------------------------------------------------
# NaN gap detection – regression tests
# ---------------------------------------------------------------------------

class TestNanGapDetection:
    """Regression tests: long NaN blocks must not produce phantom stride metrics."""

    def test_long_nan_gap_suppresses_all_stride_metrics(self):
        """Long NaN run (>15 frames) in both ankle y-columns → all stride metrics NaN."""
        fps = 30.0
        gap = 20  # > fps * 0.5 = 15 frames at 30 fps
        n = 80
        left_y = [0.9 if i % 15 < 5 else 0.4 for i in range(30)]
        left_y += [float("nan")] * gap
        left_y += [0.9 if i % 15 < 5 else 0.4 for i in range(n - 30 - gap)]
        right_y = [0.4] * n  # right foot never touches ground → no strides
        df = _make_df(n, left_y, right_y, fps=fps)

        result = analyze_strides(df, ground_threshold=0.8)

        assert result["has_long_tracking_gap"] is True
        assert result["strides"] == []
        assert math.isnan(result["stride_length"])
        assert math.isnan(result["stride_frequency"])
        assert math.isnan(result["ground_contact_ms"])

    def test_long_nan_gap_prevents_phantom_strides(self):
        """Without gap detection, ffill propagates last y-value across gap → phantom contact.
        With detection, strides list must be empty."""
        fps = 30.0
        # contact block → long gap → contact block: ffill would bridge the gap
        # and make the entire run look like one huge contact phase
        gap = 20  # > 15 frames
        left_y = [0.9] * 5 + [float("nan")] * gap + [0.9] * 5 + [0.4] * 20
        right_y = [0.4] * len(left_y)
        df = _make_df(len(left_y), left_y, right_y, fps=fps)

        result = analyze_strides(df, ground_threshold=0.8)

        assert result["has_long_tracking_gap"] is True
        assert result["strides"] == []

    def test_long_nan_gap_right_ankle_flagged(self):
        """Long NaN run in right_ankle_y also triggers the flag."""
        fps = 30.0
        gap = 20
        n = 80
        right_y = [0.9 if i % 15 < 5 else 0.4 for i in range(30)]
        right_y += [float("nan")] * gap
        right_y += [0.9 if i % 15 < 5 else 0.4 for i in range(n - 30 - gap)]
        left_y = [0.4] * n
        df = _make_df(n, left_y, right_y, fps=fps)

        result = analyze_strides(df, ground_threshold=0.8)

        assert result["has_long_tracking_gap"] is True

    def test_short_nan_gap_does_not_flag(self):
        """NaN run shorter than threshold does not set has_long_tracking_gap."""
        fps = 30.0
        df = _gait_pattern(n_strides=3, fps=fps)
        # Inject 5-frame gap — well below fps * 0.5 = 15 at 30 fps
        df = df.copy()
        df.loc[20:24, "left_ankle_y"] = float("nan")

        result = analyze_strides(df, ground_threshold=0.8)

        assert result["has_long_tracking_gap"] is False

    def test_explicit_max_gap_frames_overrides_default(self):
        """Passing max_gap_frames=25 allows a 20-frame gap without flagging."""
        fps = 30.0
        gap = 20  # would trigger at default threshold (fps * 0.5 = 15)
        n = 80
        left_y = [0.9 if i % 15 < 5 else 0.4 for i in range(30)]
        left_y += [float("nan")] * gap
        left_y += [0.9 if i % 15 < 5 else 0.4 for i in range(n - 30 - gap)]
        right_y = [0.4] * n
        df = _make_df(n, left_y, right_y, fps=fps)

        result = analyze_strides(df, ground_threshold=0.8, max_gap_frames=25)

        assert result["has_long_tracking_gap"] is False

    def test_long_gap_only_affects_gapped_foot(self):
        """Long NaN gap in left ankle suppresses left strides; right foot strides are kept."""
        fps = 30.0
        gap = 20
        contact, swing = 5, 10
        stride_block = [0.9] * contact + [0.4] * swing  # 15 frames per stride

        left_y = stride_block * 2 + [float("nan")] * gap + stride_block * 2
        # Right foot: clean gait for the full duration
        n = len(left_y)
        right_y = (stride_block * ((n // 15) + 1))[:n]
        right_x = [0.2 + 0.02 * (i // 15) for i in range(n)]

        df = _make_df(n, left_y, right_y, right_x=right_x, fps=fps)
        result = analyze_strides(df, ground_threshold=0.8)

        assert result["has_long_tracking_gap"] is True
        left_strides = [s for s in result["strides"] if s["foot"] == "left"]
        right_strides = [s for s in result["strides"] if s["foot"] == "right"]
        assert left_strides == []
        assert len(right_strides) > 0


# ---------------------------------------------------------------------------
# Plausibility bounds
# ---------------------------------------------------------------------------

class TestAnalyzeStridesBounds:
    def test_in_bounds_result_not_flagged(self):
        """Normal gait within all bounds does not set has_implausible_metric."""
        df = _gait_pattern(n_strides=3, fps=30.0, contact_frames=5, swing_frames=10, x_step=0.1)
        # calibration_factor=10 → stride_length = 0.1 * 10 = 1.0 m (within [0.1, 3.5])
        result = analyze_strides(df, calibration_factor=10.0, ground_threshold=0.8)
        assert result["has_implausible_metric"] is False
        assert not math.isnan(result["stride_length"])

    def test_stride_length_too_small_set_to_nan(self):
        """stride_length < 0.1 m is out of bounds → NaN + flag."""
        # x_step=0.001, calibration_factor=1 → stride_length = 0.001 m < 0.1
        df = _gait_pattern(n_strides=3, fps=30.0, contact_frames=5, swing_frames=10, x_step=0.001)
        result = analyze_strides(df, calibration_factor=1.0, ground_threshold=0.8)
        assert math.isnan(result["stride_length"])
        assert result["has_implausible_metric"] is True

    def test_stride_length_too_large_set_to_nan(self):
        """stride_length > 3.5 m is out of bounds → NaN + flag."""
        # x_step=0.5, calibration_factor=10 → stride_length = 5.0 m > 3.5
        df = _gait_pattern(n_strides=3, fps=30.0, contact_frames=5, swing_frames=10, x_step=0.5)
        result = analyze_strides(df, calibration_factor=10.0, ground_threshold=0.8)
        assert math.isnan(result["stride_length"])
        assert result["has_implausible_metric"] is True

    def test_stride_frequency_too_low_set_to_nan(self):
        """Very slow stride frequency < 0.5 Hz → NaN + flag.

        At fps=30, contact=5, swing=100 → stride_duration = 105/30 ≈ 3.5 s
        → frequency ≈ 0.286 Hz < 0.5 Hz.
        """
        df = _gait_pattern(n_strides=2, fps=30.0, contact_frames=5, swing_frames=100, x_step=0.1)
        result = analyze_strides(df, calibration_factor=10.0, ground_threshold=0.8)
        assert math.isnan(result["stride_frequency"])
        assert result["has_implausible_metric"] is True

    def test_stride_frequency_too_high_set_to_nan(self):
        """Very fast stride frequency > 5.0 Hz → NaN + flag.

        At fps=30, contact=1, swing=1 → stride_duration = 2/30 ≈ 0.067 s
        → frequency ≈ 15 Hz > 5.0 Hz.
        """
        df = _gait_pattern(n_strides=4, fps=30.0, contact_frames=1, swing_frames=1, x_step=0.1)
        result = analyze_strides(df, calibration_factor=1.0, ground_threshold=0.8)
        assert math.isnan(result["stride_frequency"])
        assert result["has_implausible_metric"] is True

    def test_empty_df_has_implausible_false(self):
        """Empty DataFrame returns has_implausible_metric=False (no bounds triggered)."""
        result = analyze_strides(pd.DataFrame())
        assert result["has_implausible_metric"] is False
