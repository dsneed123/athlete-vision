"""Body angle calculations: drive phase, hip extension, and forward lean."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


_TOTAL_YARDS = 40.0


def _detect_pushoff_frames(y_series: pd.Series, ground_threshold: float) -> list[int]:
    """Return frame indices where foot leaves ground (last frame of each contact phase).

    In MediaPipe normalised coordinates y increases downward, so a high y value
    means the ankle is close to the ground.  A push-off is the final frame of a
    contiguous ground-contact run before the foot enters the swing phase.
    """
    on_ground = (y_series >= ground_threshold).to_numpy()
    pushoffs: list[int] = []
    in_contact = False
    for i, val in enumerate(on_ground):
        if val and not in_contact:
            in_contact = True
        elif not val and in_contact:
            in_contact = False
            pushoffs.append(i - 1)
    return pushoffs


def analyze_angles(
    df: pd.DataFrame,
    total_distance_yards: float = _TOTAL_YARDS,
    ground_threshold: float | None = None,
) -> dict:
    """Calculate body angle metrics from pose keypoint data.

    Parameters
    ----------
    df:
        DataFrame produced by ``PoseEstimator.process_video()``.  Must contain
        hip, knee, ankle, and shoulder keypoint columns for both sides.
    total_distance_yards:
        Total sprint distance in yards.  Used to convert transition frame numbers
        to yard markers.  Default ``40`` (40-yard dash).
    ground_threshold:
        Ankle y-value (0–1, MediaPipe normalised) at or above which the foot is
        considered in ground contact.  If ``None``, computed as the 80th percentile
        of combined ankle y values, matching ``analyze_strides`` convention.

    Returns
    -------
    dict
        ``drive_phase_angle``     – average torso angle from horizontal (degrees)
                                    over the first ~25 % of frames (≈ first 10 yards).
        ``hip_extension``         – maximum hip-extension angle (degrees) recorded
                                    at push-off across all detected strides.
        ``forward_lean_angle``    – average forward-lean angle from vertical (degrees)
                                    over the first ~35 % of frames (≈ first 15 yards).
        ``transition_point_yards``– yard marker where athlete transitions from
                                    acceleration (lean > 20°) to upright running
                                    (lean < 15°).  ``nan`` when not detected.
    """
    empty: dict = {
        "drive_phase_angle": float("nan"),
        "hip_extension": float("nan"),
        "forward_lean_angle": float("nan"),
        "transition_point_yards": float("nan"),
    }

    if df.empty:
        return empty

    required = [
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y",
        "left_shoulder_x", "left_shoulder_y",
        "right_shoulder_x", "right_shoulder_y",
        "left_knee_x", "left_knee_y",
        "right_knee_x", "right_knee_y",
        "left_ankle_y", "right_ankle_y",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    n = len(df)

    # ------------------------------------------------------------------
    # Bilateral centre-of-body landmarks (ffill/bfill fills sparse gaps)
    # ------------------------------------------------------------------
    lhip_x = df["left_hip_x"].ffill().bfill().values
    lhip_y = df["left_hip_y"].ffill().bfill().values
    rhip_x = df["right_hip_x"].ffill().bfill().values
    rhip_y = df["right_hip_y"].ffill().bfill().values
    lsho_x = df["left_shoulder_x"].ffill().bfill().values
    lsho_y = df["left_shoulder_y"].ffill().bfill().values
    rsho_x = df["right_shoulder_x"].ffill().bfill().values
    rsho_y = df["right_shoulder_y"].ffill().bfill().values

    hip_cx = (lhip_x + rhip_x) / 2.0
    hip_cy = (lhip_y + rhip_y) / 2.0
    sho_cx = (lsho_x + rsho_x) / 2.0
    sho_cy = (lsho_y + rsho_y) / 2.0

    # Torso vector in physical space (y-axis flipped: positive = upward)
    dx_abs = np.abs(sho_cx - hip_cx)   # horizontal extent
    dy_up = hip_cy - sho_cy             # positive when shoulder is above hip

    # Per-frame angles (atan2 using the flipped-y torso vector)
    horiz_angles = np.degrees(np.arctan2(dy_up, dx_abs))  # from horizontal
    vert_angles = np.degrees(np.arctan2(dx_abs, dy_up))   # from vertical (forward lean)

    # ------------------------------------------------------------------
    # Adaptive ground threshold
    # ------------------------------------------------------------------
    if ground_threshold is None:
        combined_y = pd.concat([
            df["left_ankle_y"].dropna(),
            df["right_ankle_y"].dropna(),
        ])
        ground_threshold = float(combined_y.quantile(0.80)) if len(combined_y) else 0.8

    # ------------------------------------------------------------------
    # 1. Drive phase angle — average over first ~25 % of frames
    # ------------------------------------------------------------------
    drive_end = max(1, int(n * 0.25))
    drive_phase_angle = float(np.nanmean(horiz_angles[:drive_end]))

    # ------------------------------------------------------------------
    # 3. Forward lean angle — average over first ~35 % of frames
    # ------------------------------------------------------------------
    lean_end = max(1, int(n * 0.35))
    forward_lean_angle = float(np.nanmean(vert_angles[:lean_end]))

    # ------------------------------------------------------------------
    # 4. Transition point — acceleration → upright running
    #    Hysteresis: acceleration flagged above 20°, transition below 15°
    # ------------------------------------------------------------------
    transition_frame: int | None = None
    was_accelerating = False
    for i in range(n):
        angle = vert_angles[i]
        if math.isnan(angle):
            continue
        if angle > 20.0:
            was_accelerating = True
        elif was_accelerating and angle < 15.0:
            transition_frame = i
            break

    transition_point_yards = (
        (transition_frame / n) * total_distance_yards
        if transition_frame is not None
        else float("nan")
    )

    # ------------------------------------------------------------------
    # 2. Hip extension at push-off — max angle across all detected strides
    # ------------------------------------------------------------------
    lknee_x = df["left_knee_x"].ffill().bfill().values
    lknee_y = df["left_knee_y"].ffill().bfill().values
    rknee_x = df["right_knee_x"].ffill().bfill().values
    rknee_y = df["right_knee_y"].ffill().bfill().values

    extension_angles: list[float] = []

    for hip_x_arr, hip_y_arr, knee_x_arr, knee_y_arr, ankle_col in (
        (lhip_x, lhip_y, lknee_x, lknee_y, "left_ankle_y"),
        (rhip_x, rhip_y, rknee_x, rknee_y, "right_ankle_y"),
    ):
        ankle_y = df[ankle_col].ffill().bfill()
        pushoffs = _detect_pushoff_frames(ankle_y, ground_threshold)

        for fi in pushoffs:
            if fi >= n:
                continue
            hx, hy = hip_x_arr[fi], hip_y_arr[fi]
            kx, ky = knee_x_arr[fi], knee_y_arr[fi]
            sx, sy = sho_cx[fi], sho_cy[fi]

            if any(math.isnan(v) for v in (hx, hy, kx, ky, sx, sy)):
                continue

            # Angle at hip between thigh (hip→knee) and torso (hip→shoulder)
            kv_x, kv_y = kx - hx, ky - hy   # hip-to-knee
            tv_x, tv_y = sx - hx, sy - hy   # hip-to-shoulder
            dot = kv_x * tv_x + kv_y * tv_y
            mag = math.sqrt(kv_x ** 2 + kv_y ** 2) * math.sqrt(tv_x ** 2 + tv_y ** 2)
            if mag < 1e-9:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))
            extension_angles.append(angle)

    hip_extension = float(max(extension_angles)) if extension_angles else float("nan")

    return {
        "drive_phase_angle": drive_phase_angle,
        "hip_extension": hip_extension,
        "forward_lean_angle": forward_lean_angle,
        "transition_point_yards": transition_point_yards,
    }
