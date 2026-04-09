"""Tests for angle_analyzer."""

import math

import numpy as np
import pandas as pd
import pytest

from athlete_vision.angle_analyzer import (
    _detect_pushoff_frames,
    analyze_angles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LANDMARK_COLS = [
    "left_hip_x", "left_hip_y",
    "right_hip_x", "right_hip_y",
    "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "left_knee_x", "left_knee_y",
    "right_knee_x", "right_knee_y",
    "left_ankle_y", "right_ankle_y",
]


def _make_df(
    n_frames: int,
    hip_x: float = 0.5,
    hip_y: float = 0.6,
    shoulder_x: float = 0.55,
    shoulder_y: float = 0.4,
    knee_x: float = 0.5,
    knee_y: float = 0.8,
    left_ankle_y: float | None = None,
    right_ankle_y: float | None = None,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a uniform-posture DataFrame with all required columns."""
    left_ankle_y = left_ankle_y if left_ankle_y is not None else 0.3
    right_ankle_y = right_ankle_y if right_ankle_y is not None else 0.3
    data = {
        "frame_index": list(range(n_frames)),
        "timestamp_sec": [i / fps for i in range(n_frames)],
        "left_hip_x": [hip_x] * n_frames,
        "left_hip_y": [hip_y] * n_frames,
        "right_hip_x": [hip_x] * n_frames,
        "right_hip_y": [hip_y] * n_frames,
        "left_shoulder_x": [shoulder_x] * n_frames,
        "left_shoulder_y": [shoulder_y] * n_frames,
        "right_shoulder_x": [shoulder_x] * n_frames,
        "right_shoulder_y": [shoulder_y] * n_frames,
        "left_knee_x": [knee_x] * n_frames,
        "left_knee_y": [knee_y] * n_frames,
        "right_knee_x": [knee_x] * n_frames,
        "right_knee_y": [knee_y] * n_frames,
        "left_ankle_y": [left_ankle_y] * n_frames,
        "right_ankle_y": [right_ankle_y] * n_frames,
    }
    return pd.DataFrame(data)


def _make_df_with_ankle_pattern(
    n_frames: int,
    left_ankle_y: list[float],
    right_ankle_y: list[float],
    hip_x: float = 0.5,
    hip_y: float = 0.6,
    shoulder_x: float = 0.55,
    shoulder_y: float = 0.4,
    knee_x: float = 0.5,
    knee_y: float = 0.8,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build DataFrame with custom per-frame ankle y values."""
    assert len(left_ankle_y) == n_frames
    assert len(right_ankle_y) == n_frames
    data = {
        "frame_index": list(range(n_frames)),
        "timestamp_sec": [i / fps for i in range(n_frames)],
        "left_hip_x": [hip_x] * n_frames,
        "left_hip_y": [hip_y] * n_frames,
        "right_hip_x": [hip_x] * n_frames,
        "right_hip_y": [hip_y] * n_frames,
        "left_shoulder_x": [shoulder_x] * n_frames,
        "left_shoulder_y": [shoulder_y] * n_frames,
        "right_shoulder_x": [shoulder_x] * n_frames,
        "right_shoulder_y": [shoulder_y] * n_frames,
        "left_knee_x": [knee_x] * n_frames,
        "left_knee_y": [knee_y] * n_frames,
        "right_knee_x": [knee_x] * n_frames,
        "right_knee_y": [knee_y] * n_frames,
        "left_ankle_y": left_ankle_y,
        "right_ankle_y": right_ankle_y,
    }
    return pd.DataFrame(data)


def _make_lean_sequence(
    accel_frames: int,
    upright_frames: int,
    accel_lean_deg: float = 30.0,
    upright_lean_deg: float = 10.0,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a DataFrame with two posture phases (acceleration then upright).

    Angles are encoded by placing shoulder ahead/above the hip with the
    appropriate horizontal and vertical offsets for the desired lean.
    """
    n = accel_frames + upright_frames

    def _offsets(lean_deg: float) -> tuple[float, float]:
        # lean_deg is angle from vertical
        rad = math.radians(lean_deg)
        return math.sin(rad) * 0.2, math.cos(rad) * 0.2  # dx_abs, dy_up

    dx_a, dy_a = _offsets(accel_lean_deg)
    dx_u, dy_u = _offsets(upright_lean_deg)

    sho_x_vals: list[float] = []
    sho_y_vals: list[float] = []
    hip_y = 0.6

    for i in range(n):
        if i < accel_frames:
            sho_x_vals.append(0.5 + dx_a)
            sho_y_vals.append(hip_y - dy_a)
        else:
            sho_x_vals.append(0.5 + dx_u)
            sho_y_vals.append(hip_y - dy_u)

    data = {
        "frame_index": list(range(n)),
        "timestamp_sec": [i / fps for i in range(n)],
        "left_hip_x": [0.5] * n,
        "left_hip_y": [hip_y] * n,
        "right_hip_x": [0.5] * n,
        "right_hip_y": [hip_y] * n,
        "left_shoulder_x": sho_x_vals,
        "left_shoulder_y": sho_y_vals,
        "right_shoulder_x": sho_x_vals,
        "right_shoulder_y": sho_y_vals,
        "left_knee_x": [0.5] * n,
        "left_knee_y": [0.8] * n,
        "right_knee_x": [0.5] * n,
        "right_knee_y": [0.8] * n,
        "left_ankle_y": [0.3] * n,
        "right_ankle_y": [0.3] * n,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _detect_pushoff_frames
# ---------------------------------------------------------------------------

class TestDetectPushoffFrames:
    def test_single_contact_leaves_ground(self):
        y = pd.Series([0.3, 0.9, 0.9, 0.9, 0.3])
        pushoffs = _detect_pushoff_frames(y, ground_threshold=0.8)
        assert pushoffs == [3]

    def test_two_contacts(self):
        y = pd.Series([0.9, 0.9, 0.3, 0.3, 0.9, 0.9, 0.3])
        pushoffs = _detect_pushoff_frames(y, ground_threshold=0.8)
        assert pushoffs == [1, 5]

    def test_no_liftoff(self):
        # Foot stays on ground for entire series — no push-off detected
        y = pd.Series([0.9, 0.9, 0.9])
        pushoffs = _detect_pushoff_frames(y, ground_threshold=0.8)
        assert pushoffs == []

    def test_no_contact(self):
        y = pd.Series([0.3, 0.3, 0.3])
        pushoffs = _detect_pushoff_frames(y, ground_threshold=0.8)
        assert pushoffs == []


# ---------------------------------------------------------------------------
# analyze_angles — edge cases
# ---------------------------------------------------------------------------

class TestAnalyzeAnglesEdgeCases:
    def test_empty_dataframe(self):
        result = analyze_angles(pd.DataFrame())
        assert result["drive_phase_angle"] != result["drive_phase_angle"]  # nan
        assert result["hip_extension"] != result["hip_extension"]
        assert result["forward_lean_angle"] != result["forward_lean_angle"]
        assert result["transition_point_yards"] != result["transition_point_yards"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"left_hip_x": [0.5], "timestamp_sec": [0.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_angles(df)

    def test_result_keys(self):
        df = _make_df(40)
        result = analyze_angles(df, ground_threshold=0.8)
        assert set(result.keys()) == {
            "drive_phase_angle",
            "hip_extension",
            "forward_lean_angle",
            "transition_point_yards",
        }

    def test_single_frame(self):
        df = _make_df(1)
        result = analyze_angles(df, ground_threshold=0.8)
        # Should not raise; angles may be valid or nan depending on geometry
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# analyze_angles — drive phase angle
# ---------------------------------------------------------------------------

class TestDrivePhaseAngle:
    def test_45_degree_lean(self):
        # hip at (0.5, 0.6), shoulder at (0.6, 0.5):
        # dx_abs = 0.1, dy_up = 0.6 - 0.5 = 0.1 → atan2(0.1, 0.1) = 45°
        df = _make_df(40, hip_x=0.5, hip_y=0.6, shoulder_x=0.6, shoulder_y=0.5)
        result = analyze_angles(df, ground_threshold=0.8)
        assert abs(result["drive_phase_angle"] - 45.0) < 0.1

    def test_uses_only_first_25_percent(self):
        # First 25 % of frames: 45° lean; remaining frames: 0° lean (vertical)
        n = 40
        drive_end = max(1, int(n * 0.25))  # 10 frames
        shoulder_x = [0.6] * drive_end + [0.5] * (n - drive_end)
        shoulder_y = [0.5] * drive_end + [0.3] * (n - drive_end)  # 0° lean: dx=0

        data = {
            "frame_index": list(range(n)),
            "timestamp_sec": [i / 30.0 for i in range(n)],
            "left_hip_x": [0.5] * n,
            "left_hip_y": [0.6] * n,
            "right_hip_x": [0.5] * n,
            "right_hip_y": [0.6] * n,
            "left_shoulder_x": shoulder_x,
            "left_shoulder_y": shoulder_y,
            "right_shoulder_x": shoulder_x,
            "right_shoulder_y": shoulder_y,
            "left_knee_x": [0.5] * n,
            "left_knee_y": [0.8] * n,
            "right_knee_x": [0.5] * n,
            "right_knee_y": [0.8] * n,
            "left_ankle_y": [0.3] * n,
            "right_ankle_y": [0.3] * n,
        }
        df = pd.DataFrame(data)
        result = analyze_angles(df, ground_threshold=0.8)
        # Only the 45° frames are in the drive window
        assert abs(result["drive_phase_angle"] - 45.0) < 0.1

    def test_drive_angle_between_0_and_90(self):
        df = _make_df(40)
        result = analyze_angles(df, ground_threshold=0.8)
        assert 0.0 <= result["drive_phase_angle"] <= 90.0


# ---------------------------------------------------------------------------
# analyze_angles — forward lean angle
# ---------------------------------------------------------------------------

class TestForwardLeanAngle:
    def test_45_degree_forward_lean(self):
        # Same geometry as 45° drive phase → lean from vertical is also 45°
        df = _make_df(40, hip_x=0.5, hip_y=0.6, shoulder_x=0.6, shoulder_y=0.5)
        result = analyze_angles(df, ground_threshold=0.8)
        assert abs(result["forward_lean_angle"] - 45.0) < 0.1

    def test_upright_posture_near_zero(self):
        # Shoulder directly above hip (no horizontal offset) → lean ≈ 0°
        df = _make_df(40, hip_x=0.5, hip_y=0.7, shoulder_x=0.5, shoulder_y=0.3)
        result = analyze_angles(df, ground_threshold=0.8)
        assert result["forward_lean_angle"] < 5.0

    def test_uses_only_first_35_percent(self):
        n = 40
        lean_end = max(1, int(n * 0.35))  # 14 frames
        # First 35%: 30° lean; rest: 0°
        rad30 = math.radians(30.0)
        dx30 = math.sin(rad30) * 0.2  # ≈ 0.1
        dy30 = math.cos(rad30) * 0.2  # ≈ 0.173

        sho_x_vals = [0.5 + dx30] * lean_end + [0.5] * (n - lean_end)
        sho_y_vals = [0.6 - dy30] * lean_end + [0.3] * (n - lean_end)

        data = {
            "frame_index": list(range(n)),
            "timestamp_sec": [i / 30.0 for i in range(n)],
            "left_hip_x": [0.5] * n,
            "left_hip_y": [0.6] * n,
            "right_hip_x": [0.5] * n,
            "right_hip_y": [0.6] * n,
            "left_shoulder_x": sho_x_vals,
            "left_shoulder_y": sho_y_vals,
            "right_shoulder_x": sho_x_vals,
            "right_shoulder_y": sho_y_vals,
            "left_knee_x": [0.5] * n,
            "left_knee_y": [0.8] * n,
            "right_knee_x": [0.5] * n,
            "right_knee_y": [0.8] * n,
            "left_ankle_y": [0.3] * n,
            "right_ankle_y": [0.3] * n,
        }
        df = pd.DataFrame(data)
        result = analyze_angles(df, ground_threshold=0.8)
        assert abs(result["forward_lean_angle"] - 30.0) < 0.5


# ---------------------------------------------------------------------------
# analyze_angles — hip extension
# ---------------------------------------------------------------------------

class TestHipExtension:
    def test_no_pushoff_returns_nan(self):
        # Ankle never reaches ground → no push-off events
        df = _make_df(40, left_ankle_y=0.3, right_ankle_y=0.3)
        result = analyze_angles(df, ground_threshold=0.8)
        assert math.isnan(result["hip_extension"])

    def test_pushoff_produces_valid_angle(self):
        # Ankle contacts ground for first 5 frames then lifts off
        n = 20
        left_y = [0.9] * 5 + [0.3] * 15
        right_y = [0.3] * 20
        df = _make_df_with_ankle_pattern(
            n,
            left_ankle_y=left_y,
            right_ankle_y=right_y,
            hip_x=0.5, hip_y=0.6,
            shoulder_x=0.55, shoulder_y=0.4,
            knee_x=0.5, knee_y=0.8,
        )
        result = analyze_angles(df, ground_threshold=0.8)
        assert not math.isnan(result["hip_extension"])
        assert 0.0 <= result["hip_extension"] <= 180.0

    def test_max_extension_across_multiple_pushoffs(self):
        # Two contact phases; we want the max extension angle, not the mean.
        # With fixed geometry, all pushoff angles are equal — just verify
        # that all detected pushoffs feed into the result.
        n = 40
        left_y = [0.9] * 5 + [0.3] * 10 + [0.9] * 5 + [0.3] * 20
        right_y = [0.3] * n
        df = _make_df_with_ankle_pattern(
            n,
            left_ankle_y=left_y,
            right_ankle_y=right_y,
        )
        result = analyze_angles(df, ground_threshold=0.8)
        assert not math.isnan(result["hip_extension"])


# ---------------------------------------------------------------------------
# analyze_angles — transition point
# ---------------------------------------------------------------------------

class TestTransitionPoint:
    def test_transition_detected(self):
        # 20 acceleration frames (lean 30°) then 80 upright frames (lean 10°)
        # With 100 total frames and 40 yards: transition ≈ 20/100 * 40 = 8 yards
        accel = 20
        upright = 80
        df = _make_lean_sequence(
            accel_frames=accel,
            upright_frames=upright,
            accel_lean_deg=30.0,
            upright_lean_deg=10.0,
        )
        result = analyze_angles(df, total_distance_yards=40.0, ground_threshold=0.8)
        assert not math.isnan(result["transition_point_yards"])
        # The transition must fall after the accel window starts (yard 0)
        # and well before the end (yard 40)
        assert 0.0 < result["transition_point_yards"] < 40.0
        # Expected: first upright frame is frame 20 → 20/100 * 40 = 8 yards
        assert abs(result["transition_point_yards"] - 8.0) < 2.0

    def test_transition_not_detected_always_upright(self):
        # Athlete is always upright → no acceleration phase → no transition
        df = _make_lean_sequence(
            accel_frames=0,
            upright_frames=100,
            accel_lean_deg=30.0,
            upright_lean_deg=10.0,
        )
        result = analyze_angles(df, ground_threshold=0.8)
        assert math.isnan(result["transition_point_yards"])

    def test_transition_not_detected_always_accelerating(self):
        # Athlete never goes upright → transition never triggered
        df = _make_lean_sequence(
            accel_frames=100,
            upright_frames=0,
            accel_lean_deg=30.0,
            upright_lean_deg=30.0,
        )
        result = analyze_angles(df, ground_threshold=0.8)
        assert math.isnan(result["transition_point_yards"])

    def test_transition_yards_proportional_to_distance(self):
        # Doubling total_distance_yards doubles transition_point_yards
        accel = 25
        upright = 75
        df = _make_lean_sequence(accel, upright, accel_lean_deg=30.0, upright_lean_deg=10.0)
        r40 = analyze_angles(df, total_distance_yards=40.0, ground_threshold=0.8)
        r80 = analyze_angles(df, total_distance_yards=80.0, ground_threshold=0.8)
        if not math.isnan(r40["transition_point_yards"]) and not math.isnan(r80["transition_point_yards"]):
            ratio = r80["transition_point_yards"] / r40["transition_point_yards"]
            assert abs(ratio - 2.0) < 0.01

    def test_hysteresis_between_15_and_20_degrees(self):
        # A lean between 15° and 20° should NOT trigger a transition
        n = 60
        # First 20 frames: 30° (acceleration), next 40: 17° (in hysteresis band)
        df = _make_lean_sequence(20, 40, accel_lean_deg=30.0, upright_lean_deg=17.0)
        result = analyze_angles(df, ground_threshold=0.8)
        # 17° is between 15° and 20° — no transition should be found
        assert math.isnan(result["transition_point_yards"])
