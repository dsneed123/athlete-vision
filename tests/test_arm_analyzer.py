"""Tests for arm_analyzer."""

import math

import numpy as np
import pandas as pd
import pytest

from athlete_vision.arm_analyzer import (
    _arm_amplitude,
    _cross_body_ratio,
    analyze_arm_swing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_frames: int,
    left_wrist_x: list[float],
    right_wrist_x: list[float],
    left_shoulder_x: list[float] | None = None,
    right_shoulder_x: list[float] | None = None,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Build a minimal DataFrame with wrist and shoulder x columns."""
    left_shoulder_x = left_shoulder_x or [0.3] * n_frames
    right_shoulder_x = right_shoulder_x or [0.7] * n_frames
    return pd.DataFrame(
        {
            "frame_index": list(range(n_frames)),
            "timestamp_sec": [i / fps for i in range(n_frames)],
            "left_wrist_x": left_wrist_x,
            "right_wrist_x": right_wrist_x,
            "left_shoulder_x": left_shoulder_x,
            "right_shoulder_x": right_shoulder_x,
        }
    )


def _symmetric_swing(
    n_frames: int = 60,
    amplitude: float = 0.1,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Synthesise a perfectly symmetric arm swing pattern."""
    t = np.linspace(0, 2 * np.pi, n_frames)
    left_x = (0.3 + amplitude * np.sin(t)).tolist()
    right_x = (0.7 + amplitude * np.sin(t + np.pi)).tolist()
    return _make_df(n_frames, left_x, right_x, fps=fps)


def _asymmetric_swing(
    n_frames: int = 60,
    left_amplitude: float = 0.1,
    right_amplitude: float = 0.05,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Synthesise an asymmetric arm swing where left > right."""
    t = np.linspace(0, 2 * np.pi, n_frames)
    left_x = (0.3 + left_amplitude * np.sin(t)).tolist()
    right_x = (0.7 + right_amplitude * np.sin(t + np.pi)).tolist()
    return _make_df(n_frames, left_x, right_x, fps=fps)


# ---------------------------------------------------------------------------
# _arm_amplitude
# ---------------------------------------------------------------------------

class TestArmAmplitude:
    def test_known_amplitude(self):
        wrist_x = np.array([0.2, 0.3, 0.4, 0.3, 0.2])
        shoulder_x = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        amp = _arm_amplitude(wrist_x, shoulder_x)
        # relative_x = [-0.1, 0.0, 0.1, 0.0, -0.1] → range = 0.2
        assert abs(amp - 0.2) < 1e-9

    def test_zero_amplitude(self):
        wrist_x = np.array([0.3, 0.3, 0.3])
        shoulder_x = np.array([0.3, 0.3, 0.3])
        amp = _arm_amplitude(wrist_x, shoulder_x)
        assert abs(amp) < 1e-9

    def test_amplitude_always_nonnegative(self):
        wrist_x = np.random.rand(50)
        shoulder_x = np.random.rand(50)
        amp = _arm_amplitude(wrist_x, shoulder_x)
        assert amp >= 0.0


# ---------------------------------------------------------------------------
# _cross_body_ratio
# ---------------------------------------------------------------------------

class TestCrossBodyRatio:
    def test_no_cross_body_left(self):
        # Left wrist always to the left of the midline (0.5) → no cross-body
        midline_x = np.array([0.5, 0.5, 0.5, 0.5])
        wrist_x = np.array([0.1, 0.2, 0.3, 0.4])  # relative_x < 0
        ratio = _cross_body_ratio(wrist_x, midline_x, "left")
        assert ratio == 0.0

    def test_full_cross_body_left(self):
        # Left wrist always past midline → 100 % cross-body
        midline_x = np.array([0.5, 0.5, 0.5])
        wrist_x = np.array([0.6, 0.7, 0.8])  # relative_x > 0 throughout
        ratio = _cross_body_ratio(wrist_x, midline_x, "left")
        assert ratio == 1.0

    def test_no_cross_body_right(self):
        # Right wrist always to the right of the midline → no cross-body
        midline_x = np.array([0.5, 0.5, 0.5])
        wrist_x = np.array([0.6, 0.7, 0.8])  # relative_x > 0
        ratio = _cross_body_ratio(wrist_x, midline_x, "right")
        assert ratio == 0.0

    def test_full_cross_body_right(self):
        # Right wrist always to the left of midline → 100 % cross-body
        midline_x = np.array([0.5, 0.5, 0.5])
        wrist_x = np.array([0.1, 0.2, 0.3])  # relative_x < 0
        ratio = _cross_body_ratio(wrist_x, midline_x, "right")
        assert ratio == 1.0

    def test_partial_cross_body(self):
        midline_x = np.array([0.5, 0.5, 0.5, 0.5])
        # 2 of 4 frames cross the midline → ratio = 0.5
        wrist_x = np.array([0.3, 0.4, 0.6, 0.7])
        ratio = _cross_body_ratio(wrist_x, midline_x, "left")
        assert abs(ratio - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# analyze_arm_swing – edge cases
# ---------------------------------------------------------------------------

class TestAnalyzeArmSwingEdgeCases:
    def test_empty_dataframe(self):
        result = analyze_arm_swing(pd.DataFrame())
        assert math.isnan(result["arm_swing_symmetry"])
        assert math.isnan(result["left_arm_amplitude"])
        assert math.isnan(result["right_arm_amplitude"])
        assert math.isnan(result["left_cross_body_ratio"])
        assert math.isnan(result["right_cross_body_ratio"])
        assert result["cross_body_swing"] is False

    def test_missing_column_raises(self):
        df = pd.DataFrame(
            {"timestamp_sec": [0.0], "left_wrist_x": [0.3], "right_wrist_x": [0.7]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_arm_swing(df)

    def test_result_keys(self):
        df = _symmetric_swing()
        result = analyze_arm_swing(df)
        expected_keys = {
            "arm_swing_symmetry",
            "left_arm_amplitude",
            "right_arm_amplitude",
            "left_cross_body_ratio",
            "right_cross_body_ratio",
            "cross_body_swing",
        }
        assert set(result.keys()) == expected_keys

    def test_single_frame(self):
        df = _make_df(1, [0.3], [0.7])
        result = analyze_arm_swing(df)
        # Single frame → amplitude = 0 for both arms, symmetry = 100
        assert abs(result["left_arm_amplitude"]) < 1e-9
        assert abs(result["right_arm_amplitude"]) < 1e-9
        assert abs(result["arm_swing_symmetry"] - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# analyze_arm_swing – symmetry score
# ---------------------------------------------------------------------------

class TestAnalyzeArmSwingSymmetry:
    def test_perfect_symmetry(self):
        # Both arms have the same amplitude → score = 100
        df = _symmetric_swing(amplitude=0.1)
        result = analyze_arm_swing(df)
        assert abs(result["arm_swing_symmetry"] - 100.0) < 1.0

    def test_partial_symmetry(self):
        # Left 0.1, right 0.05 → ratio = 50 % difference → score ≈ 50
        df = _asymmetric_swing(left_amplitude=0.1, right_amplitude=0.05)
        result = analyze_arm_swing(df)
        # |0.1 - 0.05| / 0.1 * 100 = 50 → symmetry = 50
        assert abs(result["arm_swing_symmetry"] - 50.0) < 2.0

    def test_symmetry_range(self):
        df = _asymmetric_swing()
        result = analyze_arm_swing(df)
        score = result["arm_swing_symmetry"]
        assert 0.0 <= score <= 100.0

    def test_one_arm_static(self):
        # Right wrist perfectly still → symmetry = 0
        n = 30
        t = np.linspace(0, 2 * np.pi, n)
        left_x = (0.3 + 0.1 * np.sin(t)).tolist()
        right_x = [0.7] * n  # amplitude = 0
        df = _make_df(n, left_x, right_x)
        result = analyze_arm_swing(df)
        assert abs(result["arm_swing_symmetry"]) < 2.0  # near zero

    def test_amplitudes_stored(self):
        df = _asymmetric_swing(left_amplitude=0.1, right_amplitude=0.05)
        result = analyze_arm_swing(df)
        # Left amplitude should be larger than right
        assert result["left_arm_amplitude"] > result["right_arm_amplitude"]


# ---------------------------------------------------------------------------
# analyze_arm_swing – cross-body detection
# ---------------------------------------------------------------------------

class TestAnalyzeArmSwingCrossBody:
    def test_no_cross_body_flag(self):
        df = _symmetric_swing(amplitude=0.05)
        result = analyze_arm_swing(df)
        # Straight-back swing → no cross-body
        assert result["cross_body_swing"] is False

    def test_cross_body_flag_left(self):
        # Left wrist always past midline (0.5) → cross-body for left arm
        # Midline = (left_shoulder + right_shoulder) / 2 = (0.3 + 0.7) / 2 = 0.5
        n = 30
        left_x = [0.6] * n   # wrist well past midline
        right_x = [0.7] * n
        df = _make_df(n, left_x, right_x)
        result = analyze_arm_swing(df)
        assert result["cross_body_swing"] is True
        assert result["left_cross_body_ratio"] == 1.0

    def test_cross_body_flag_right(self):
        # Right wrist always left of midline (0.5) → cross-body for right arm
        n = 30
        left_x = [0.3] * n
        right_x = [0.4] * n   # wrist left of midline (0.5)
        df = _make_df(n, left_x, right_x)
        result = analyze_arm_swing(df)
        assert result["cross_body_swing"] is True
        assert result["right_cross_body_ratio"] == 1.0

    def test_cross_body_threshold(self):
        # Only 5 % of frames cross-body → below 10 % threshold → flag stays False
        n = 40
        # 2 frames past midline (0.5), 38 frames well below → 2/40 = 5 %
        left_x = [0.6] * 2 + [0.2] * 38
        right_x = [0.7] * n
        df = _make_df(n, left_x, right_x)
        result = analyze_arm_swing(df)
        assert result["cross_body_swing"] is False
