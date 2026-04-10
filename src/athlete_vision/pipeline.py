"""Full analysis pipeline: folder processing, CSV output, quality flagging, and summary."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .angle_analyzer import analyze_angles
from .arm_analyzer import analyze_arm_swing
from .pose_estimator import PoseEstimator
from .stride_analyzer import analyze_strides
from .velocity_analyzer import analyze_velocity

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Critical joints used for confidence and tracking-loss checks
_CRITICAL_JOINTS = frozenset(
    {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
)

# Data-quality thresholds
_MIN_CONFIDENCE = 0.85           # Average visibility for critical joints
_MIN_FRAMES = 100                # Minimum frame count
_MAX_TRACKING_LOSS_RATIO = 0.10  # Max fraction of frames with any critical joint NaN

# Standard landscape aspect ratios (width / height).
# Portrait (9/16 ≈ 0.56) is intentionally absent — it flags as non-standard.
_STANDARD_RATIOS = (16 / 9, 4 / 3, 3 / 2)
_ASPECT_TOLERANCE = 0.15         # Allowed relative deviation from each standard ratio

# Plausible 40-yard dash range (seconds)
_TIME_MIN = 3.5
_TIME_MAX = 6.5

# Pose-plausibility threshold: more than this fraction of frames failing any
# biomechanical check triggers the IMPLAUSIBLE_POSE flag.
_MAX_IMPLAUSIBLE_RATIO = 0.05

# Canonical CSV column order
_OUTPUT_COLUMNS = [
    "athlete_id",
    "video_filename",
    "forty_time",
    "stride_length",
    "stride_frequency",
    "ground_contact_ms",
    "drive_phase_angle",
    "hip_extension",
    "arm_swing_symmetry",
    "forward_lean_angle",
    "transition_point_yards",
    "peak_velocity_mph",
    "data_quality",
]


def _get_video_aspect_ratio(video_path: Path) -> float | None:
    """Return width/height aspect ratio of a video, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if height == 0:
        return None
    return width / height


def _check_data_quality(df: pd.DataFrame, video_path: Path) -> str:
    """Return ``'REVIEW'`` when any quality criterion fails, else ``'OK'``.

    Criteria (any one triggers REVIEW):

    * Fewer than 100 frames — video too short for reliable analysis.
    * More than 10 % of frames have a critical joint (hip / knee / ankle)
      with NaN x-coordinate — tracking was lost too often.
    * Average visibility on critical joints falls below 85 % — low
      MediaPipe confidence on the most important landmarks.
    * Video aspect ratio is not within 15 % of 16:9, 4:3, or 3:2 —
      portrait or unusual framing may distort pose geometry.
    """
    if len(df) < _MIN_FRAMES:
        return "REVIEW"

    # Tracking loss: fraction of frames where any critical joint is absent
    crit_x_cols = [f"{j}_x" for j in _CRITICAL_JOINTS if f"{j}_x" in df.columns]
    if crit_x_cols:
        loss_ratio = float(df[crit_x_cols].isna().any(axis=1).mean())
        if loss_ratio > _MAX_TRACKING_LOSS_RATIO:
            return "REVIEW"

    # Average confidence on critical joints (stored in DataFrame attrs)
    avg_conf: dict = df.attrs.get("avg_confidence", {})
    crit_confs = [
        v for k, v in avg_conf.items()
        if k in _CRITICAL_JOINTS and not math.isnan(v)
    ]
    if crit_confs and (sum(crit_confs) / len(crit_confs)) < _MIN_CONFIDENCE:
        return "REVIEW"

    # Aspect ratio check
    ratio = _get_video_aspect_ratio(video_path)
    if ratio is not None:
        if not any(
            abs(ratio - std) / std <= _ASPECT_TOLERANCE
            for std in _STANDARD_RATIOS
        ):
            return "REVIEW"

    return "OK"


def validate_pose_plausibility(df: pd.DataFrame) -> bool:
    """Return ``False`` when > 5 % of frames fail any biomechanical check.

    Normalised coordinates have y increasing downward (y=0 top, y=1 bottom),
    so a joint that is physically lower in the body has a *larger* y value.

    Checks performed on every frame (both left and right sides):

    1. **Ankle below hip**: ``ankle_y > hip_y`` — ankle must not appear above
       the hip in the frame.
    2. **Knee between hip and ankle**: ``hip_y < knee_y < ankle_y``.
    3. **Elbow between shoulder and wrist**: elbow y lies within
       ``[min(shoulder_y, wrist_y), max(shoulder_y, wrist_y)]``.
    4. **No side-crossing**: mean x of left joints must not exceed mean x of
       right joints (would indicate a mirrored / swapped pose).

    Frames where any joint involved in a check is NaN are skipped for that
    check (benefit of the doubt for missing detections).
    """
    if df.empty:
        return True

    def _fail_ratio(fail: np.ndarray, valid: np.ndarray) -> float:
        """Fraction of *valid* frames where *fail* is True."""
        n = int(valid.sum())
        return float(fail[valid].sum() / n) if n else 0.0

    # Checks 1 & 2: lower-limb vertical chain
    for side in ("left", "right"):
        hip_y = df[f"{side}_hip_y"].to_numpy(dtype=float)
        knee_y = df[f"{side}_knee_y"].to_numpy(dtype=float)
        ankle_y = df[f"{side}_ankle_y"].to_numpy(dtype=float)

        valid_ha = ~(np.isnan(hip_y) | np.isnan(ankle_y))
        valid_hka = valid_ha & ~np.isnan(knee_y)

        # Check 1: ankle below hip
        if _fail_ratio(ankle_y < hip_y, valid_ha) > _MAX_IMPLAUSIBLE_RATIO:
            return False

        # Check 2: knee vertically between hip and ankle
        knee_ok = (knee_y > hip_y) & (knee_y < ankle_y)
        if _fail_ratio(~knee_ok, valid_hka) > _MAX_IMPLAUSIBLE_RATIO:
            return False

    # Check 3: upper-limb vertical chain
    for side in ("left", "right"):
        shoulder_y = df[f"{side}_shoulder_y"].to_numpy(dtype=float)
        elbow_y = df[f"{side}_elbow_y"].to_numpy(dtype=float)
        wrist_y = df[f"{side}_wrist_y"].to_numpy(dtype=float)

        valid_sew = ~(np.isnan(shoulder_y) | np.isnan(elbow_y) | np.isnan(wrist_y))
        lower = np.minimum(shoulder_y, wrist_y)
        upper = np.maximum(shoulder_y, wrist_y)
        elbow_ok = (elbow_y >= lower) & (elbow_y <= upper)
        if _fail_ratio(~elbow_ok, valid_sew) > _MAX_IMPLAUSIBLE_RATIO:
            return False

    # Check 4: left joints must not be consistently to the right of right joints
    left_x_cols = [
        c for c in (
            "left_hip_x", "left_knee_x", "left_ankle_x",
            "left_shoulder_x", "left_elbow_x", "left_wrist_x",
        )
        if c in df.columns
    ]
    right_x_cols = [
        c for c in (
            "right_hip_x", "right_knee_x", "right_ankle_x",
            "right_shoulder_x", "right_elbow_x", "right_wrist_x",
        )
        if c in df.columns
    ]
    if left_x_cols and right_x_cols:
        left_mean = df[left_x_cols].mean(axis=1).to_numpy(dtype=float)
        right_mean = df[right_x_cols].mean(axis=1).to_numpy(dtype=float)
        valid_lr = ~(np.isnan(left_mean) | np.isnan(right_mean))
        if _fail_ratio(left_mean > right_mean, valid_lr) > _MAX_IMPLAUSIBLE_RATIO:
            return False

    return True


def _empty_row(athlete_id: str, video_filename: str) -> dict:
    """Return a row dict pre-filled with NaN metrics and REVIEW quality."""
    return {
        "athlete_id": athlete_id,
        "video_filename": video_filename,
        "forty_time": float("nan"),
        "stride_length": float("nan"),
        "stride_frequency": float("nan"),
        "ground_contact_ms": float("nan"),
        "drive_phase_angle": float("nan"),
        "hip_extension": float("nan"),
        "arm_swing_symmetry": float("nan"),
        "forward_lean_angle": float("nan"),
        "transition_point_yards": float("nan"),
        "peak_velocity_mph": float("nan"),
        "data_quality": "REVIEW",
    }


def process_video(
    video_path: Path,
    athlete_id: str,
    estimator: PoseEstimator,
    calibration_factor: float = 1.0,
) -> tuple[dict, str, str | None]:
    """Run the full analysis pipeline on one video file.

    Parameters
    ----------
    video_path:
        Path to the video file.
    athlete_id:
        Identifier written into the ``athlete_id`` CSV column.
    estimator:
        An already-initialised :class:`PoseEstimator` instance.
    calibration_factor:
        Converts one normalised x-coordinate unit to metres.  Derive via
        :func:`athlete_vision.calibration.calibrate`.  Default ``1.0`` returns
        distance metrics in normalised units.

    Returns
    -------
    tuple[dict, str, str | None]
        ``(row, status, error)`` where *status* is one of ``'ok'``,
        ``'no_pose'``, or ``'error'``, and *error* is a message string
        when *status* is ``'error'``, else ``None``.
    """
    row = _empty_row(athlete_id, video_path.name)

    try:
        df = estimator.process_video(str(video_path))

        if df.empty:
            return row, "no_pose", None

        # --- Pose plausibility (before analyzers) ---
        plausible = validate_pose_plausibility(df)

        # --- Stride metrics ---
        stride_metrics = analyze_strides(df, calibration_factor=calibration_factor)
        row["stride_length"] = stride_metrics["stride_length"]
        row["stride_frequency"] = stride_metrics["stride_frequency"]
        row["ground_contact_ms"] = stride_metrics["ground_contact_ms"]

        # --- Angle metrics ---
        angle_metrics = analyze_angles(df)
        row["drive_phase_angle"] = angle_metrics["drive_phase_angle"]
        row["hip_extension"] = angle_metrics["hip_extension"]
        row["forward_lean_angle"] = angle_metrics["forward_lean_angle"]
        row["transition_point_yards"] = angle_metrics["transition_point_yards"]

        # --- Arm swing ---
        arm_metrics = analyze_arm_swing(df)
        row["arm_swing_symmetry"] = arm_metrics["arm_swing_symmetry"]

        # --- Velocity / 40-time ---
        vel_metrics = analyze_velocity(df, calibration_factor=calibration_factor)
        row["peak_velocity_mph"] = vel_metrics["peak_velocity_mph"]
        forty = vel_metrics["forty_time"]
        if not math.isnan(forty) and _TIME_MIN <= forty <= _TIME_MAX:
            row["forty_time"] = forty

        # --- Data quality ---
        quality = _check_data_quality(df, video_path)
        if not plausible:
            quality = f"{quality}|IMPLAUSIBLE_POSE"
        row["data_quality"] = quality

        return row, "ok", None

    except Exception as exc:
        return row, "error", str(exc)


def run_pipeline(
    video_dir: Path,
    output_csv: Path,
    model_complexity: int = 1,
    athlete_id: str | None = None,
    calibration_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Process all videos in *video_dir* through the full analysis pipeline.

    Parameters
    ----------
    video_dir:
        Directory containing video files (.mp4, .mov, .avi, .mkv).
    output_csv:
        Destination path for the output CSV.
    model_complexity:
        MediaPipe model complexity: 0 (lite), 1 (full), 2 (heavy).
    athlete_id:
        Athlete identifier embedded in every CSV row.  Defaults to the
        video filename stem when ``None``.
    calibration_factor:
        Converts one normalised x-coordinate unit to metres.  Derive via
        :func:`athlete_vision.calibration.calibrate`.  Default ``1.0`` returns
        distance metrics in normalised units.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame (one row per video) and a stats dict with keys
        ``processed``, ``flagged``, and ``failed``.

    Raises
    ------
    FileNotFoundError
        When no supported video files are found in *video_dir*.
    """
    video_dir = Path(video_dir)
    video_files = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    stats: dict = {"processed": 0, "flagged": 0, "failed": 0}
    rows: list[dict] = []

    with PoseEstimator(model_complexity=model_complexity) as estimator:
        for video_path in video_files:
            aid = athlete_id if athlete_id is not None else video_path.stem
            print(f"  {video_path.name} ...", end="", flush=True)

            row, status, error = process_video(
                video_path, aid, estimator, calibration_factor=calibration_factor
            )

            if status == "error":
                print(f" [error: {error}]")
                stats["failed"] += 1
            elif status == "no_pose":
                print(" [no_pose]")
                stats["failed"] += 1
            else:
                quality = row["data_quality"]
                print(f" [{quality}]")
                stats["processed"] += 1
                if quality.split("|")[0] == "REVIEW":
                    stats["flagged"] += 1

            rows.append(row)

    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    df.to_csv(output_csv, index=False)
    return df, stats


def print_pipeline_summary(df: pd.DataFrame, stats: dict) -> None:
    """Print a human-readable pipeline run summary to stdout."""
    print("\n=== Pipeline Summary ===")
    print(f"  Files processed : {stats['processed']}")
    print(f"  Flagged (REVIEW): {stats['flagged']}")
    print(f"  Failed          : {stats['failed']}")
    print(f"  Total           : {len(df)}")
    implausible = int(df["data_quality"].str.contains("IMPLAUSIBLE_POSE", na=False).sum())
    if implausible:
        print(f"  Implausible pose: {implausible}")

    # 40-time distribution
    times = pd.to_numeric(df["forty_time"], errors="coerce").dropna()
    if not times.empty:
        print("\n--- 40-Yard Time Distribution ---")
        bins = [3.5, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.5]
        labels = [
            "3.5–4.0", "4.0–4.2", "4.2–4.4", "4.4–4.6",
            "4.6–4.8", "4.8–5.0", "5.0–5.5", "5.5–6.5",
        ]
        counts, _ = np.histogram(times.values, bins=bins)
        bar_max = int(counts.max()) if counts.max() > 0 else 1
        for label, count in zip(labels, counts):
            bar = "█" * int(20 * count / bar_max)
            print(f"  {label}s  {bar:<20} {count}")
        print(
            f"  Mean: {times.mean():.2f}s  "
            f"Min: {times.min():.2f}s  "
            f"Max: {times.max():.2f}s"
        )

    # Peak velocity distribution
    vels = pd.to_numeric(df["peak_velocity_mph"], errors="coerce").dropna()
    if not vels.empty:
        print("\n--- Peak Velocity Distribution (mph) ---")
        print(f"  Min  : {vels.min():.1f} mph")
        print(f"  Mean : {vels.mean():.1f} mph")
        print(f"  Max  : {vels.max():.1f} mph")
