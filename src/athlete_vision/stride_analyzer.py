"""Stride analysis: length, frequency, and ground contact time from pose keypoints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _detect_contact_phases(
    y_series: pd.Series, threshold: float
) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) frame-index pairs for ground-contact phases.

    In MediaPipe's normalised coordinate system y increases downward, so a high
    y value means the ankle is close to the ground.  A contact phase is a
    contiguous run of frames where ankle_y >= threshold.
    """
    on_ground = (y_series >= threshold).to_numpy()
    phases: list[tuple[int, int]] = []
    in_contact = False
    start = 0
    for i, val in enumerate(on_ground):
        if val and not in_contact:
            in_contact = True
            start = i
        elif not val and in_contact:
            in_contact = False
            phases.append((start, i - 1))
    if in_contact:
        phases.append((start, len(y_series) - 1))
    return phases


def _strides_for_foot(
    df: pd.DataFrame,
    foot: str,
    calibration_factor: float,
    ground_threshold: float,
) -> list[dict]:
    """Extract per-stride metrics for one foot ('left' or 'right').

    A stride is the interval from one ground-contact touchdown to the next
    ground-contact touchdown of the same foot.
    """
    ankle_y = df[f"{foot}_ankle_y"].ffill().bfill()
    ankle_x = df[f"{foot}_ankle_x"].ffill().bfill()
    timestamps = df["timestamp_sec"]

    phases = _detect_contact_phases(ankle_y, ground_threshold)
    if len(phases) < 2:
        return []

    strides: list[dict] = []
    for i in range(len(phases) - 1):
        # Stride boundaries: touchdown i → touchdown i+1
        td_current = phases[i][0]
        td_next = phases[i + 1][0]

        t_start = float(timestamps.iloc[td_current])
        t_end = float(timestamps.iloc[td_next])
        stride_duration_s = t_end - t_start

        if stride_duration_s <= 0:
            continue

        # Stride length: horizontal displacement between successive touchdowns
        x_delta = abs(float(ankle_x.iloc[td_next]) - float(ankle_x.iloc[td_current]))
        stride_length_m = x_delta * calibration_factor

        # Ground contact time for the current stance phase
        contact_start = phases[i][0]
        contact_end = phases[i][1]
        ground_contact_s = (
            float(timestamps.iloc[contact_end])
            - float(timestamps.iloc[contact_start])
        )
        ground_contact_ms = ground_contact_s * 1000.0

        strides.append(
            {
                "foot": foot,
                "stride_index": i,
                "start_frame": td_current,
                "end_frame": td_next,
                "stride_length_m": stride_length_m,
                "stride_frequency_hz": 1.0 / stride_duration_s,
                "ground_contact_ms": ground_contact_ms,
            }
        )

    return strides


def analyze_strides(
    df: pd.DataFrame,
    calibration_factor: float = 1.0,
    ground_threshold: float | None = None,
) -> dict:
    """Analyse stride metrics from a pose keypoint DataFrame.

    Parameters
    ----------
    df:
        DataFrame produced by ``PoseEstimator.process_video()``.  Must contain
        ``left_ankle_x``, ``left_ankle_y``, ``right_ankle_x``,
        ``right_ankle_y``, and ``timestamp_sec`` columns.
    calibration_factor:
        Converts normalised x-coordinate units to metres.  Derive it as::

            calibration_factor = known_distance_metres / known_distance_in_normalised_units

        For example, if a 40-yard line (36.576 m) spans 0.9 of the frame
        width, ``calibration_factor = 36.576 / 0.9 ≈ 40.6``.
        Default ``1.0`` returns stride length in normalised units.
    ground_threshold:
        Ankle y-value (0–1, MediaPipe normalised) at or above which the foot
        is considered in ground contact.  Higher y = lower in frame = closer
        to ground.  If ``None``, computed as the 80th percentile of the
        combined ankle y values so it adapts to each clip's camera angle.

    Returns
    -------
    dict
        ``strides``           – list of per-stride dicts (one entry per detected stride)
        ``stride_length``     – average stride length in metres (or normalised units)
        ``stride_frequency``  – average stride frequency in strides per second
        ``ground_contact_ms`` – average ground contact time in milliseconds
    """
    empty_result: dict = {
        "strides": [],
        "stride_length": float("nan"),
        "stride_frequency": float("nan"),
        "ground_contact_ms": float("nan"),
    }

    if df.empty:
        return empty_result

    required = [
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "timestamp_sec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if ground_threshold is None:
        combined_y = pd.concat(
            [df["left_ankle_y"].dropna(), df["right_ankle_y"].dropna()]
        )
        ground_threshold = float(combined_y.quantile(0.80)) if len(combined_y) else 0.8

    all_strides: list[dict] = []
    for foot in ("left", "right"):
        all_strides.extend(
            _strides_for_foot(df, foot, calibration_factor, ground_threshold)
        )

    if not all_strides:
        return empty_result

    return {
        "strides": all_strides,
        "stride_length": float(np.mean([s["stride_length_m"] for s in all_strides])),
        "stride_frequency": float(
            np.mean([s["stride_frequency_hz"] for s in all_strides])
        ),
        "ground_contact_ms": float(
            np.mean([s["ground_contact_ms"] for s in all_strides])
        ),
    }
