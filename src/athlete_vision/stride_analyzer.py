"""Stride analysis: length, frequency, and ground contact time from pose keypoints."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

# Default gap threshold: NaN runs longer than this fraction of a second are
# considered a "long tracking gap" and suppress stride metrics for that foot.
_GAP_THRESHOLD_SECONDS = 0.5

# Plausibility bounds for aggregate stride metrics
_STRIDE_LENGTH_MIN = 0.1    # metres
_STRIDE_LENGTH_MAX = 3.5    # metres
_GROUND_CONTACT_MS_MIN = 50.0   # milliseconds
_GROUND_CONTACT_MS_MAX = 500.0  # milliseconds
_STRIDE_FREQUENCY_MIN = 0.5    # Hz
_STRIDE_FREQUENCY_MAX = 5.0    # Hz


def _has_long_nan_run(series: pd.Series, max_gap: int) -> bool:
    """Return True if any contiguous NaN run in *series* exceeds *max_gap* frames."""
    run = 0
    for val in series.isna():
        if val:
            run += 1
            if run > max_gap:
                return True
        else:
            run = 0
    return False


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
    max_gap_frames: int,
) -> tuple[list[dict], bool]:
    """Extract per-stride metrics for one foot ('left' or 'right').

    A stride is the interval from one ground-contact touchdown to the next
    ground-contact touchdown of the same foot.

    Returns
    -------
    tuple[list[dict], bool]
        ``(strides, has_long_gap)`` where *has_long_gap* is ``True`` when the
        ankle y-series contains a contiguous NaN run longer than
        *max_gap_frames*.  In that case *strides* is always empty, because
        filling the gap with ffill/bfill would produce phantom contact phases
        with incorrect timing.
    """
    raw_ankle_y = df[f"{foot}_ankle_y"]

    if _has_long_nan_run(raw_ankle_y, max_gap_frames):
        return [], True

    ankle_y = raw_ankle_y.ffill().bfill()
    ankle_x = df[f"{foot}_ankle_x"].ffill().bfill()
    timestamps = df["timestamp_sec"]

    phases = _detect_contact_phases(ankle_y, ground_threshold)
    if len(phases) < 2:
        return [], False

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

    return strides, False


def analyze_strides(
    df: pd.DataFrame,
    calibration_factor: float = 1.0,
    ground_threshold: float | None = None,
    max_gap_frames: int | None = None,
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
    max_gap_frames:
        Maximum number of consecutive NaN frames allowed in an ankle
        y-series before stride metrics for that foot are suppressed.  When
        ``None`` (default) the threshold is derived from the clip's frame
        rate as ``fps * 0.5`` (half a second).  Pass an explicit integer to
        override this, e.g. ``max_gap_frames=15``.

    Returns
    -------
    dict
        ``strides``                – list of per-stride dicts (one entry per detected stride)
        ``stride_length``          – average stride length in metres (or normalised units)
        ``stride_frequency``       – average stride frequency in strides per second
        ``ground_contact_ms``      – average ground contact time in milliseconds
        ``has_long_tracking_gap``  – ``True`` when a long NaN run was detected in at
                                     least one ankle y-series; affected foot's strides
                                     are excluded to prevent phantom-stride corruption
    """
    empty_result: dict = {
        "strides": [],
        "stride_length": float("nan"),
        "stride_frequency": float("nan"),
        "ground_contact_ms": float("nan"),
        "has_long_tracking_gap": False,
        "has_implausible_metric": False,
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

    if max_gap_frames is None:
        ts_diffs = df["timestamp_sec"].diff().dropna()
        median_interval = float(ts_diffs.median()) if len(ts_diffs) > 0 else 0.0
        fps = 1.0 / median_interval if median_interval > 0 else 30.0
        max_gap_frames = max(1, int(fps * _GAP_THRESHOLD_SECONDS))

    if ground_threshold is None:
        combined_y = pd.concat(
            [df["left_ankle_y"].dropna(), df["right_ankle_y"].dropna()]
        )
        ground_threshold = float(combined_y.quantile(0.80)) if len(combined_y) else 0.8

    all_strides: list[dict] = []
    has_long_gap = False
    for foot in ("left", "right"):
        strides, foot_has_gap = _strides_for_foot(
            df, foot, calibration_factor, ground_threshold, max_gap_frames
        )
        all_strides.extend(strides)
        has_long_gap = has_long_gap or foot_has_gap

    if not all_strides:
        return {**empty_result, "has_long_tracking_gap": has_long_gap}

    result: dict = {
        "strides": all_strides,
        "stride_length": float(np.mean([s["stride_length_m"] for s in all_strides])),
        "stride_frequency": float(
            np.mean([s["stride_frequency_hz"] for s in all_strides])
        ),
        "ground_contact_ms": float(
            np.mean([s["ground_contact_ms"] for s in all_strides])
        ),
        "has_long_tracking_gap": has_long_gap,
        "has_implausible_metric": False,
    }

    # Post-calculation plausibility bounds
    _bounds: list[tuple[str, float, float]] = [
        ("stride_length", _STRIDE_LENGTH_MIN, _STRIDE_LENGTH_MAX),
        ("ground_contact_ms", _GROUND_CONTACT_MS_MIN, _GROUND_CONTACT_MS_MAX),
        ("stride_frequency", _STRIDE_FREQUENCY_MIN, _STRIDE_FREQUENCY_MAX),
    ]
    for metric, lo, hi in _bounds:
        value = result[metric]
        if not math.isnan(value) and not (lo <= value <= hi):
            logging.warning(
                "stride_analyzer: %s=%.4g is outside plausible range [%g, %g]; setting to NaN",
                metric, value, lo, hi,
            )
            result[metric] = float("nan")
            result["has_implausible_metric"] = True

    return result
