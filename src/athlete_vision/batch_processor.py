"""Batch processing pipeline for 40-yard dash videos."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .pose_estimator import PoseEstimator
from .stride_analyzer import analyze_strides

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

# Plausible 40-yard dash range (seconds)
_TIME_MIN = 3.5
_TIME_MAX = 6.5


def _angle_between(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """Return the angle at vertex B (degrees) formed by points A–B–C."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax ** 2 + bay ** 2) * math.sqrt(bcx ** 2 + bcy ** 2)
    if mag < 1e-9:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def calculate_angles(df: pd.DataFrame) -> dict:
    """Compute per-frame joint angles and return their frame-averages.

    Angles computed (both sides):
    * knee angle – hip → knee → ankle
    * hip angle  – shoulder → hip → knee

    Returns
    -------
    dict
        Keys: left_knee_angle_deg, right_knee_angle_deg,
              left_hip_angle_deg, right_hip_angle_deg
    """
    result: dict = {}
    n = len(df)

    for side in ("left", "right"):
        hip_x = df[f"{side}_hip_x"].values
        hip_y = df[f"{side}_hip_y"].values
        knee_x = df[f"{side}_knee_x"].values
        knee_y = df[f"{side}_knee_y"].values
        ankle_x = df[f"{side}_ankle_x"].values
        ankle_y = df[f"{side}_ankle_y"].values
        shoulder_x = df[f"{side}_shoulder_x"].values
        shoulder_y = df[f"{side}_shoulder_y"].values

        knee_angles = np.empty(n)
        hip_angles = np.empty(n)
        for i in range(n):
            knee_angles[i] = _angle_between(
                hip_x[i], hip_y[i],
                knee_x[i], knee_y[i],
                ankle_x[i], ankle_y[i],
            )
            hip_angles[i] = _angle_between(
                shoulder_x[i], shoulder_y[i],
                hip_x[i], hip_y[i],
                knee_x[i], knee_y[i],
            )

        result[f"{side}_knee_angle_deg"] = float(np.nanmean(knee_angles))
        result[f"{side}_hip_angle_deg"] = float(np.nanmean(hip_angles))

    return result


def calculate_velocity(df: pd.DataFrame) -> dict:
    """Estimate horizontal velocity from hip centre-of-mass movement.

    Velocity is in normalised-coordinate units per second (not m/s) — use as
    a relative metric only.

    Returns
    -------
    dict
        Keys: peak_velocity_norm, avg_velocity_norm
    """
    if df.empty or len(df) < 2:
        return {"peak_velocity_norm": float("nan"), "avg_velocity_norm": float("nan")}

    hip_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    timestamps = df["timestamp_sec"].values

    dt = np.diff(timestamps)
    dx = np.abs(np.diff(hip_x.values))
    with np.errstate(invalid="ignore"):
        velocities = dx / (dt + 1e-9)

    return {
        "peak_velocity_norm": float(np.nanmax(velocities)),
        "avg_velocity_norm": float(np.nanmean(velocities)),
    }


def calculate_arm_swing(df: pd.DataFrame) -> dict:
    """Compute arm swing amplitude from wrist position relative to shoulder.

    Amplitude is the peak-to-peak range of (wrist_y − shoulder_y) over the
    clip, in normalised units.

    Returns
    -------
    dict
        Keys: left_arm_swing_amplitude, right_arm_swing_amplitude,
              arm_swing_amplitude (mean of both sides)
    """
    result: dict = {}
    total = 0.0
    sides_with_data = 0

    for side in ("left", "right"):
        wrist_y = df[f"{side}_wrist_y"].values
        shoulder_y = df[f"{side}_shoulder_y"].values
        relative_y = wrist_y - shoulder_y
        amp = float(np.nanmax(relative_y) - np.nanmin(relative_y))
        result[f"{side}_arm_swing_amplitude"] = amp
        if not math.isnan(amp):
            total += amp
            sides_with_data += 1

    result["arm_swing_amplitude"] = (
        total / sides_with_data if sides_with_data else float("nan")
    )
    return result


def _extract_40_time(df: pd.DataFrame) -> Optional[float]:
    """Estimate 40-yard dash time from horizontal centre-of-mass movement.

    Finds the window of sustained hip displacement and uses its duration as a
    proxy for race time.  Returns None when the estimate falls outside the
    plausible range [3.5, 6.5] s or when data is insufficient.
    """
    if df.empty or len(df) < 10:
        return None

    hip_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    timestamps = df["timestamp_sec"].values

    if len(hip_x) < 2:
        return None

    dx = np.abs(np.diff(hip_x.ffill().values))
    nonzero = dx[dx > 0]
    if len(nonzero) == 0:
        return None

    threshold = float(np.percentile(nonzero, 30))
    moving = np.where(dx > threshold)[0]
    if len(moving) < 2:
        return None

    start_frame = int(moving[0])
    end_frame = int(moving[-1])
    end_ts_idx = min(end_frame + 1, len(timestamps) - 1)
    elapsed = float(timestamps[end_ts_idx] - timestamps[start_frame])

    return elapsed if _TIME_MIN <= elapsed <= _TIME_MAX else None


def _lookup_metadata(video_name: str, metadata_by_id: dict) -> dict:
    """Return the metadata record whose video_id is a prefix of the filename stem."""
    stem = Path(video_name).stem
    for video_id, meta in metadata_by_id.items():
        if stem.startswith(str(video_id)):
            return meta
    # Fallback: direct filename match
    for meta in metadata_by_id.values():
        if meta.get("filename") == video_name:
            return meta
    return {}


def process_single_video(
    video_path: Path,
    metadata: dict,
    model_complexity: int = 1,
) -> dict:
    """Run the full analysis pipeline on one video file.

    Parameters
    ----------
    video_path:
        Path to the video file.
    metadata:
        Metadata dict (from metadata.json) with optional keys:
        athlete_name, known_time (seconds).
    model_complexity:
        MediaPipe model complexity: 0 (lite), 1 (full), 2 (heavy).

    Returns
    -------
    dict
        All extracted metrics plus ``status`` (ok / no_pose / flagged / error)
        and ``error`` (None unless status is error).
    """
    result: dict = {
        "video": video_path.name,
        "athlete": metadata.get("athlete_name") or metadata.get("athlete"),
        "known_40_time": metadata.get("known_time") or metadata.get("time_40_yards"),
        "status": "ok",
        "error": None,
        "extracted_40_time": float("nan"),
        "stride_length_m": float("nan"),
        "stride_frequency_hz": float("nan"),
        "ground_contact_ms": float("nan"),
        "left_knee_angle_deg": float("nan"),
        "right_knee_angle_deg": float("nan"),
        "left_hip_angle_deg": float("nan"),
        "right_hip_angle_deg": float("nan"),
        "peak_velocity_norm": float("nan"),
        "avg_velocity_norm": float("nan"),
        "left_arm_swing_amplitude": float("nan"),
        "right_arm_swing_amplitude": float("nan"),
        "arm_swing_amplitude": float("nan"),
    }

    try:
        with PoseEstimator(model_complexity=model_complexity) as estimator:
            df = estimator.process_video(str(video_path))

        if df.empty:
            result["status"] = "no_pose"
            return result

        stride_metrics = analyze_strides(df)
        result["stride_length_m"] = stride_metrics["stride_length"]
        result["stride_frequency_hz"] = stride_metrics["stride_frequency"]
        result["ground_contact_ms"] = stride_metrics["ground_contact_ms"]

        result.update(calculate_angles(df))
        result.update(calculate_velocity(df))
        result.update(calculate_arm_swing(df))

        t = _extract_40_time(df)
        result["extracted_40_time"] = t if t is not None else float("nan")

        # Flag implausible stride lengths for review
        sl = result["stride_length_m"]
        if not math.isnan(sl) and (sl < 0.05 or sl > 3.0):
            result["status"] = "flagged"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def batch_process(
    video_dir: Path,
    output_csv: Path,
    model_complexity: int = 1,
) -> pd.DataFrame:
    """Process every video in *video_dir* through the full pipeline.

    Reads ``metadata.json`` from *video_dir* when present to obtain known
    40-yard dash times and athlete names.  Writes results to *output_csv*
    and returns the DataFrame.

    Raises
    ------
    FileNotFoundError
        When no video files are found in *video_dir*.
    """
    video_dir = Path(video_dir)
    metadata_path = video_dir / "metadata.json"

    metadata_by_id: dict = {}
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text())
            if isinstance(raw, dict):
                metadata_by_id = raw
            elif isinstance(raw, list):
                for rec in raw:
                    key = rec.get("video_id") or rec.get("filename", "")
                    if key:
                        metadata_by_id[key] = rec
        except (json.JSONDecodeError, AttributeError):
            pass

    video_files = sorted(
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    results: list[dict] = []
    for video_path in video_files:
        meta = _lookup_metadata(video_path.name, metadata_by_id)
        print(f"  Processing {video_path.name} ...", end="", flush=True)
        result = process_single_video(video_path, meta, model_complexity)
        print(f" [{result['status']}]")
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a full batch-processing summary to stdout."""
    total = len(df)
    ok = int((df["status"] == "ok").sum())
    failed = int((df["status"] == "error").sum())
    no_pose = int((df["status"] == "no_pose").sum())
    flagged = int((df["status"] == "flagged").sum())

    print("\n=== Batch Processing Summary ===")
    print(f"  Total videos  : {total}")
    print(f"  Successful    : {ok}")
    print(f"  No pose       : {no_pose}")
    print(f"  Flagged       : {flagged}")
    print(f"  Failed        : {failed}")

    # 40-time distribution histogram
    times = pd.to_numeric(df["extracted_40_time"], errors="coerce").dropna()
    if not times.empty:
        print("\n--- Extracted 40-Yard Time Distribution ---")
        bins = [3.5, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.5]
        labels = [
            "3.5–4.0", "4.0–4.2", "4.2–4.4", "4.4–4.6",
            "4.6–4.8", "4.8–5.0", "5.0–5.5", "5.5–6.5",
        ]
        counts, _ = np.histogram(times, bins=bins)
        bar_max = int(counts.max()) if counts.max() > 0 else 1
        for label, count in zip(labels, counts):
            bar = "█" * int(20 * count / bar_max)
            print(f"  {label}s  {bar:<20} {count}")

    # Peak velocity distribution
    vels = pd.to_numeric(
        df.get("peak_velocity_norm", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if not vels.empty:
        print("\n--- Peak Velocity (normalised units/s) ---")
        print(f"  Min  : {vels.min():.4f}")
        print(f"  Mean : {vels.mean():.4f}")
        print(f"  Max  : {vels.max():.4f}")

    # Stride averages
    sl_vals = pd.to_numeric(
        df.get("stride_length_m", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    sf_vals = pd.to_numeric(
        df.get("stride_frequency_hz", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if not sl_vals.empty or not sf_vals.empty:
        print("\n--- Stride Averages (all athletes) ---")
        if not sl_vals.empty:
            print(f"  Avg stride length    : {sl_vals.mean():.4f} (normalised units)")
        if not sf_vals.empty:
            print(f"  Avg stride frequency : {sf_vals.mean():.3f} Hz")

    # Accuracy report
    known_num = pd.to_numeric(df["known_40_time"], errors="coerce")
    extracted_num = pd.to_numeric(df["extracted_40_time"], errors="coerce")
    has_ground_truth = known_num.notna() & extracted_num.notna()
    if has_ground_truth.any():
        sub = df[has_ground_truth].copy()
        sub = sub.assign(_known=known_num[has_ground_truth], _extracted=extracted_num[has_ground_truth])
        sub = sub.dropna(subset=["_known", "_extracted"])
        if not sub.empty:
            sub = sub.assign(_err=(sub["_extracted"] - sub["_known"]).abs())
            mean_err = sub["_err"].mean()
            print(f"\n--- Accuracy Report ({len(sub)} video(s) with ground truth) ---")
            print(f"  Mean absolute error : {mean_err:.3f} s")
            for _, row in sub.iterrows():
                name = str(row["video"])[:40]
                print(
                    f"  {name:<40}  known={row['_known']:.2f}s"
                    f"  extracted={row['_extracted']:.2f}s"
                    f"  err={row['_err']:.2f}s"
                )


def generate_html_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a self-contained HTML report with embedded distribution charts.

    Requires matplotlib.  All chart images are embedded as base64 PNGs so the
    output file has no external dependencies.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`batch_process` (or loaded from its CSV).
    output_path:
        Destination ``.html`` file path.

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for HTML report generation. "
            "Install it with: pip install matplotlib"
        ) from exc

    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    charts: list[tuple[str, str]] = []  # (b64_png, title)

    # Chart 1 — 40-time histogram
    times = pd.to_numeric(df["extracted_40_time"], errors="coerce").dropna()
    if not times.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            times,
            bins=[3.5, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.5],
            color="#1f77b4",
            edgecolor="white",
        )
        ax.set_title("Extracted 40-Yard Dash Time Distribution")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Count")
        charts.append((_fig_to_b64(fig), "40-Yard Dash Time Distribution"))

    # Chart 2 — peak velocity histogram
    vels = pd.to_numeric(
        df.get("peak_velocity_norm", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if not vels.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(vels, bins=15, color="#2ca02c", edgecolor="white")
        ax.set_title("Peak Velocity Distribution (normalised units/s)")
        ax.set_xlabel("Peak Velocity")
        ax.set_ylabel("Count")
        charts.append((_fig_to_b64(fig), "Peak Velocity Distribution"))

    # Chart 3 — stride length histogram
    sl = pd.to_numeric(
        df.get("stride_length_m", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if not sl.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sl, bins=15, color="#ff7f0e", edgecolor="white")
        ax.set_title("Stride Length Distribution (normalised units)")
        ax.set_xlabel("Stride Length")
        ax.set_ylabel("Count")
        charts.append((_fig_to_b64(fig), "Stride Length Distribution"))

    # Chart 4 — accuracy scatter (extracted vs known)
    known_num = pd.to_numeric(df["known_40_time"], errors="coerce")
    extracted_num = pd.to_numeric(df["extracted_40_time"], errors="coerce")
    valid = known_num.notna() & extracted_num.notna()
    if valid.any():
        kn = known_num[valid]
        ex = extracted_num[valid]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(kn, ex, color="#d62728", zorder=3)
        lo = min(kn.min(), ex.min()) - 0.1
        hi = max(kn.max(), ex.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect accuracy")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title("Extracted vs Known 40-Yard Dash Time")
        ax.set_xlabel("Known Time (s)")
        ax.set_ylabel("Extracted Time (s)")
        ax.legend()
        charts.append((_fig_to_b64(fig), "Accuracy: Extracted vs Known"))

    # Build summary stats
    total = len(df)
    ok = int((df["status"] == "ok").sum())
    failed = int((df["status"] == "error").sum())
    flagged = int((df["status"] == "flagged").sum())

    chart_html = "\n".join(
        f'<div class="chart"><h2>{title}</h2>'
        f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>'
        for b64, title in charts
    )

    # Per-video table rows
    status_cls = {"ok": "ok", "error": "error", "flagged": "flagged", "no_pose": "warn"}
    table_rows: list[str] = []
    for _, row in df.iterrows():
        cls = status_cls.get(str(row.get("status", "")), "")

        def _fmt(val, fmt=".2f") -> str:
            try:
                f = float(val)
                return f"{f:{fmt}}" if not math.isnan(f) else "—"
            except (TypeError, ValueError):
                return "—"

        table_rows.append(
            f'<tr class="{cls}">'
            f"<td>{row.get('video', '')}</td>"
            f"<td>{row.get('athlete') or '—'}</td>"
            f"<td>{_fmt(row.get('extracted_40_time'))}s</td>"
            f"<td>{_fmt(row.get('known_40_time'))}s</td>"
            f"<td>{_fmt(row.get('stride_length_m'), '.4f')}</td>"
            f"<td>{row.get('status', '')}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Athlete Vision — Dataset Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px;
            background: #f5f5f5; color: #333; }}
    h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
    .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
    .stat {{ background: #fff; border-radius: 8px; padding: 16px 24px;
             box-shadow: 0 2px 4px rgba(0,0,0,.1); text-align: center; }}
    .stat .value {{ font-size: 2em; font-weight: bold; color: #1f77b4; }}
    .stat .label {{ font-size: .9em; color: #666; }}
    .chart {{ background: #fff; border-radius: 8px; padding: 20px;
              margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
    .chart h2 {{ margin-top: 0; color: #444; }}
    .chart img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 4px rgba(0,0,0,.1); margin-top: 20px; }}
    th {{ background: #1a1a2e; color: #fff; padding: 10px 12px;
          text-align: left; font-weight: 600; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
    tr.ok {{ background: #f0fff0; }}
    tr.error {{ background: #fff0f0; }}
    tr.flagged {{ background: #fffbe6; }}
    tr.warn {{ background: #f0f8ff; }}
  </style>
</head>
<body>
  <h1>Athlete Vision — Dataset Report</h1>
  <div class="stats">
    <div class="stat"><div class="value">{total}</div><div class="label">Total Videos</div></div>
    <div class="stat"><div class="value">{ok}</div><div class="label">Successful</div></div>
    <div class="stat"><div class="value">{flagged}</div><div class="label">Flagged</div></div>
    <div class="stat"><div class="value">{failed}</div><div class="label">Failed</div></div>
  </div>
  {chart_html}
  <div class="chart">
    <h2>Per-Video Results</h2>
    <table>
      <thead>
        <tr>
          <th>Video</th><th>Athlete</th><th>Extracted Time</th>
          <th>Known Time</th><th>Stride Length</th><th>Status</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
