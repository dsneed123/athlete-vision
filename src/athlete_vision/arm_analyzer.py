"""Arm swing symmetry analysis: amplitude, symmetry score, and swing direction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _arm_amplitude(wrist_x: np.ndarray, shoulder_x: np.ndarray) -> float:
    """Return peak-to-peak swing amplitude for one arm.

    Amplitude = (max forward position - max backward position) of the wrist
    relative to the shoulder in the x-axis (horizontal).  In MediaPipe's
    coordinate system, x increases left-to-right so forward/backward swings
    appear as positive and negative x offsets relative to the shoulder.
    """
    relative_x = wrist_x - shoulder_x
    x_min = float(np.nanmin(relative_x))
    x_max = float(np.nanmax(relative_x))
    return x_max - x_min


def _cross_body_ratio(
    wrist_x: np.ndarray,
    midline_x: np.ndarray,
    side: str,
) -> float:
    """Return the fraction of frames where the wrist crosses the body midline.

    Cross-body arm swing is detected when the wrist moves past the body
    midline (centre between both shoulders) toward the opposite side of the
    body.  Returns a ratio in [0, 1] — values above 0.1 suggest cross-body
    tendency (bad form).

    Parameters
    ----------
    wrist_x:
        Wrist x-coordinates (normalised, 0–1).
    midline_x:
        Body midline x-coordinates (normalised, 0–1), typically the mean of
        both shoulder x values at each frame.
    side:
        ``'left'`` or ``'right'``.  Controls which direction counts as
        cross-body (left wrist crossing to the right of the midline, or vice
        versa).
    """
    relative_x = wrist_x - midline_x
    if side == "left":
        # Left wrist crossing to the right of the midline = cross-body
        cross = relative_x > 0.0
    else:
        # Right wrist crossing to the left of the midline = cross-body
        cross = relative_x < 0.0

    valid = ~np.isnan(relative_x)
    if not valid.any():
        return float("nan")
    return float(np.sum(cross & valid) / np.sum(valid))


def analyze_arm_swing(
    df: pd.DataFrame,
) -> dict:
    """Analyse arm swing symmetry and form from pose keypoint data.

    Parameters
    ----------
    df:
        DataFrame produced by ``PoseEstimator.process_video()``.  Must contain
        ``left_wrist_x``, ``left_wrist_y``, ``right_wrist_x``,
        ``right_wrist_y``, ``left_shoulder_x``, ``right_shoulder_x``,
        and ``timestamp_sec`` columns.

    Returns
    -------
    dict
        ``arm_swing_symmetry``   – symmetry score 0–100 (100 = perfectly symmetric).
        ``left_arm_amplitude``   – peak-to-peak x amplitude for the left arm
                                   (normalised units).
        ``right_arm_amplitude``  – peak-to-peak x amplitude for the right arm
                                   (normalised units).
        ``left_cross_body_ratio``  – fraction of frames where the left wrist
                                     crosses the body midline (0–1).
        ``right_cross_body_ratio`` – fraction of frames where the right wrist
                                     crosses the body midline (0–1).
        ``cross_body_swing``     – ``True`` when either arm crosses the midline
                                   in more than 10 % of frames (bad form flag).
    """
    empty: dict = {
        "arm_swing_symmetry": float("nan"),
        "left_arm_amplitude": float("nan"),
        "right_arm_amplitude": float("nan"),
        "left_cross_body_ratio": float("nan"),
        "right_cross_body_ratio": float("nan"),
        "cross_body_swing": False,
    }

    if df.empty:
        return empty

    required = [
        "left_wrist_x",
        "right_wrist_x",
        "left_shoulder_x",
        "right_shoulder_x",
        "timestamp_sec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Fill sparse NaN gaps before computing
    lwrist_x = df["left_wrist_x"].ffill().bfill().values
    rwrist_x = df["right_wrist_x"].ffill().bfill().values
    lsho_x = df["left_shoulder_x"].ffill().bfill().values
    rsho_x = df["right_shoulder_x"].ffill().bfill().values

    left_amp = _arm_amplitude(lwrist_x, lsho_x)
    right_amp = _arm_amplitude(rwrist_x, rsho_x)

    # Symmetry score: 100 = identical amplitudes, 0 = one arm is stationary
    if math.isnan(left_amp) or math.isnan(right_amp):
        symmetry = float("nan")
    else:
        max_amp = max(left_amp, right_amp)
        if max_amp < 1e-9:
            symmetry = 100.0
        else:
            symmetry = 100.0 - abs(left_amp - right_amp) / max_amp * 100.0

    # Body midline: centre between both shoulders at each frame
    midline_x = (lsho_x + rsho_x) / 2.0

    left_cross = _cross_body_ratio(lwrist_x, midline_x, "left")
    right_cross = _cross_body_ratio(rwrist_x, midline_x, "right")

    cross_body_swing = False
    if not math.isnan(left_cross) and left_cross > 0.10:
        cross_body_swing = True
    if not math.isnan(right_cross) and right_cross > 0.10:
        cross_body_swing = True

    return {
        "arm_swing_symmetry": symmetry,
        "left_arm_amplitude": left_amp,
        "right_arm_amplitude": right_amp,
        "left_cross_body_ratio": left_cross,
        "right_cross_body_ratio": right_cross,
        "cross_body_swing": cross_body_swing,
    }
