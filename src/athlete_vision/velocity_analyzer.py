"""Velocity analysis: peak velocity (mph) and 40-yard dash time extraction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import FORTY_TIME_MAX, FORTY_TIME_MIN

# Default calibration: 40 yards = 36.576 metres.  Without a real calibration
# the caller should supply calibration_factor derived from a known reference.
_YARDS_TO_METRES = 0.9144
_FORTY_YARDS_M = 40.0 * _YARDS_TO_METRES  # 36.576 m


def _smooth_velocities(velocities: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a centred rolling-mean smoothing to a velocity array.

    Uses ``np.convolve`` with a uniform kernel, which is equivalent to a
    causal rolling mean with edge padding (``mode='same'``).  Edges are
    computed from fewer samples so they remain unbiased.
    """
    if len(velocities) < window:
        return velocities.copy()
    kernel = np.ones(window) / window
    return np.convolve(velocities, kernel, mode="same")


def _detect_movement_window(hip_x: pd.Series) -> tuple[int, int] | None:
    """Return (start_frame, end_frame) indices spanning the sustained run.

    The method mirrors ``batch_processor._extract_40_time``:
    1. Compute per-frame displacement.
    2. Take the 30th-percentile of nonzero displacements as the movement
       threshold (filters out sensor noise without discarding real motion).
    3. Clamp to the first and last frames where displacement exceeds that
       threshold.

    Returns ``None`` when fewer than two moving frames are found.
    """
    dx = np.abs(np.diff(hip_x.ffill().values))
    nonzero = dx[dx > 0]
    if len(nonzero) == 0:
        return None
    threshold = float(np.percentile(nonzero, 30))
    moving = np.where(dx > threshold)[0]
    if len(moving) < 2:
        return None
    return int(moving[0]), int(moving[-1])


def analyze_velocity(
    df: pd.DataFrame,
    calibration_factor: float = 1.0,
    smooth_window: int = 5,
) -> dict:
    """Calculate peak velocity (mph) and 40-yard dash time from hip movement.

    Parameters
    ----------
    df:
        DataFrame produced by ``PoseEstimator.process_video()``.  Must contain
        ``left_hip_x``, ``right_hip_x``, and ``timestamp_sec`` columns.
    calibration_factor:
        Converts one normalised x-coordinate unit to metres.  Derive as::

            calibration_factor = known_distance_metres / known_distance_in_normalised_units

        For example, if a 40-yard line (36.576 m) spans 0.9 of the frame
        width, ``calibration_factor = 36.576 / 0.9 ≈ 40.6``.
        Default ``1.0`` treats each normalised unit as one metre, which
        produces velocity in normalised-units/s rather than m/s.
    smooth_window:
        Number of frames for the rolling-mean velocity smoother.  Default 5.

    Returns
    -------
    dict
        ``peak_velocity_mph`` – peak smoothed horizontal velocity in mph.
        ``peak_velocity_ms``  – peak smoothed horizontal velocity in m/s.
        ``avg_velocity_ms``   – mean velocity over the detected movement window
                                in m/s (``nan`` when no movement detected).
        ``forty_time``        – estimated 40-yard dash time in seconds, derived
                                from the first-movement frame to the last-movement
                                frame (``nan`` when outside the plausible range
                                [FORTY_TIME_MIN, FORTY_TIME_MAX] or when data is
                                insufficient).
    """
    empty: dict = {
        "peak_velocity_mph": float("nan"),
        "peak_velocity_ms": float("nan"),
        "avg_velocity_ms": float("nan"),
        "forty_time": float("nan"),
    }

    if df.empty:
        return empty

    required = ["left_hip_x", "right_hip_x", "timestamp_sec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if len(df) < 2:
        return empty

    # Hip midpoint trajectory
    hip_x = (
        df["left_hip_x"].ffill().bfill() + df["right_hip_x"].ffill().bfill()
    ) / 2.0
    timestamps = df["timestamp_sec"].values

    # Per-frame displacement and time delta
    dt = np.diff(timestamps)
    dx = np.abs(np.diff(hip_x.values)) * calibration_factor  # metres

    # Guard against zero/negative time deltas
    with np.errstate(invalid="ignore", divide="ignore"):
        inst_velocity = np.where(dt > 1e-9, dx / dt, np.nan)  # m/s

    # Smooth with rolling window
    smoothed = _smooth_velocities(inst_velocity, window=smooth_window)

    peak_ms = float(np.nanmax(smoothed)) if len(smoothed) > 0 else float("nan")
    peak_mph = peak_ms * 2.23694  # m/s → mph

    # Average velocity over the movement window
    window = _detect_movement_window(hip_x)
    if window is not None:
        start_f, end_f = window
        # smoothed has length n-1 (diff of n points); clamp end index
        end_v = min(end_f, len(smoothed) - 1)
        window_vels = smoothed[start_f : end_v + 1]
        avg_ms = float(np.nanmean(window_vels)) if len(window_vels) > 0 else float("nan")
    else:
        avg_ms = float("nan")

    # 40-yard dash time: duration from first to last moving frame
    forty_time: float = float("nan")
    if window is not None:
        start_f, end_f = window
        end_ts_idx = min(end_f + 1, len(timestamps) - 1)
        elapsed = float(timestamps[end_ts_idx] - timestamps[start_f])
        if FORTY_TIME_MIN <= elapsed <= FORTY_TIME_MAX:
            forty_time = elapsed

    return {
        "peak_velocity_mph": peak_mph,
        "peak_velocity_ms": peak_ms,
        "avg_velocity_ms": avg_ms,
        "forty_time": forty_time,
    }
