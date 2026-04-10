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

        # --- Stride metrics ---
        stride_metrics = analyze_strides(df)
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
        vel_metrics = analyze_velocity(df)
        row["peak_velocity_mph"] = vel_metrics["peak_velocity_mph"]
        forty = vel_metrics["forty_time"]
        if not math.isnan(forty) and _TIME_MIN <= forty <= _TIME_MAX:
            row["forty_time"] = forty

        # --- Data quality ---
        row["data_quality"] = _check_data_quality(df, video_path)

        return row, "ok", None

    except Exception as exc:
        return row, "error", str(exc)


def run_pipeline(
    video_dir: Path,
    output_csv: Path,
    model_complexity: int = 1,
    athlete_id: str | None = None,
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

            row, status, error = process_video(video_path, aid, estimator)

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
                if quality == "REVIEW":
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
