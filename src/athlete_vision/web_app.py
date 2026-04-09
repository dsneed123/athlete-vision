"""Streamlit web GUI for Athlete Vision — 40-yard dash video analysis."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .angle_analyzer import analyze_angles
from .batch_processor import (
    _extract_40_time,
    calculate_angles,
    calculate_arm_swing,
    calculate_velocity,
)
from .pose_estimator import PoseEstimator
from .stride_analyzer import analyze_strides

# ─── Constants ────────────────────────────────────────────────────────────────

_ACCEPTED_TYPES = ["mp4", "mov", "avi"]

# Skeleton connections (landmark_a, landmark_b) for overlay rendering
_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_JOINTS = [
    "left_ankle", "right_ankle",
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

# BGR colours for left/right skeleton sides
_LEFT_COLOR = (0, 255, 128)
_RIGHT_COLOR = (0, 128, 255)
_VIS_THRESHOLD = 0.35

# Graded benchmarks — values use the same units as the analysis pipeline.
# stride_length_m / arm_swing_amplitude / peak_velocity_norm are in normalised
# video-coordinate units (not physical metres / mph).
_BENCHMARKS: dict[str, dict] = {
    "stride_length_m": {
        "label": "Stride Length",
        "unit": "norm",
        "elite": (0.06, 99.0),
        "good": (0.03, 0.06),
        "lower_is_better": False,
        "tip": "Normalised units — higher = longer stride relative to frame",
    },
    "stride_frequency_hz": {
        "label": "Stride Frequency",
        "unit": "Hz",
        "elite": (4.0, 99.0),
        "good": (2.5, 4.0),
        "lower_is_better": False,
        "tip": "Elite sprinters: 4.5–5.5 Hz",
    },
    "ground_contact_ms": {
        "label": "Ground Contact",
        "unit": "ms",
        "elite": (0.0, 120.0),
        "good": (120.0, 180.0),
        "lower_is_better": True,
        "tip": "Elite: < 120 ms per contact phase",
    },
    "drive_phase_angle": {
        "label": "Drive Phase Angle",
        "unit": "°",
        "elite": (20.0, 99.0),
        "good": (10.0, 20.0),
        "lower_is_better": False,
        "tip": "Angle from horizontal — higher = more forward lean in drive phase",
    },
    "hip_extension": {
        "label": "Hip Extension",
        "unit": "°",
        "elite": (120.0, 180.0),
        "good": (90.0, 120.0),
        "lower_is_better": False,
        "tip": "Max hip-extension angle at push-off; elite: 130–160°",
    },
    "arm_swing_amplitude": {
        "label": "Arm Swing",
        "unit": "norm",
        "elite": (0.12, 99.0),
        "good": (0.06, 0.12),
        "lower_is_better": False,
        "tip": "Wrist range relative to shoulder (normalised); larger = more powerful arm drive",
    },
    "forward_lean_angle": {
        "label": "Forward Lean",
        "unit": "°",
        "elite": (0.0, 35.0),
        "good": (35.0, 55.0),
        "lower_is_better": True,
        "tip": "Angle from vertical in first 35 % of run — lower = more upright later",
    },
    "transition_point_yards": {
        "label": "Transition Point",
        "unit": "yards",
        "elite": (0.0, 15.0),
        "good": (15.0, 22.0),
        "lower_is_better": True,
        "tip": "Yard at which athlete transitions from drive phase to upright running",
    },
    "peak_velocity_norm": {
        "label": "Peak Velocity",
        "unit": "norm/s",
        "elite": (0.08, 99.0),
        "good": (0.04, 0.08),
        "lower_is_better": False,
        "tip": "Peak horizontal velocity (normalised coords); elite real-world ≈ 21–23 mph",
    },
}

_GRADE_HEX = {
    "green": "#28a745",
    "yellow": "#e6a817",
    "red": "#dc3545",
    "gray": "#6c757d",
}
_GRADE_LABEL = {"green": "Elite", "yellow": "Good", "red": "Needs work", "gray": "N/A"}


# ─── Analysis pipeline ────────────────────────────────────────────────────────

def _analyze_video(
    video_path: str,
    model_complexity: int = 1,
    progress_cb=None,
) -> tuple[dict, pd.DataFrame]:
    """Run the full analysis pipeline on one video file.

    Returns (metrics_dict, pose_df).  pose_df is empty on failure.
    """
    def _prog(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)

    metrics: dict = {
        "video": Path(video_path).name,
        "status": "ok",
        "error": None,
        "extracted_40_time": float("nan"),
        "stride_length_m": float("nan"),
        "stride_frequency_hz": float("nan"),
        "ground_contact_ms": float("nan"),
        "drive_phase_angle": float("nan"),
        "forward_lean_angle": float("nan"),
        "hip_extension": float("nan"),
        "transition_point_yards": float("nan"),
        "peak_velocity_norm": float("nan"),
        "avg_velocity_norm": float("nan"),
        "left_arm_swing_amplitude": float("nan"),
        "right_arm_swing_amplitude": float("nan"),
        "arm_swing_amplitude": float("nan"),
        "arm_swing_symmetry": float("nan"),
        "left_knee_angle_deg": float("nan"),
        "right_knee_angle_deg": float("nan"),
        "left_hip_angle_deg": float("nan"),
        "right_hip_angle_deg": float("nan"),
        "avg_confidence": {},
        "strides": [],
    }

    pose_df: pd.DataFrame = pd.DataFrame()
    try:
        _prog(0.05, "Opening video and extracting pose keypoints...")
        with PoseEstimator(model_complexity=model_complexity) as est:
            pose_df = est.process_video(video_path)

        if pose_df.empty:
            metrics["status"] = "no_pose"
            return metrics, pose_df

        metrics["avg_confidence"] = pose_df.attrs.get("avg_confidence", {})

        _prog(0.35, "Analysing strides...")
        strides_result = analyze_strides(pose_df)
        metrics["stride_length_m"] = strides_result.get("stride_length", float("nan"))
        metrics["stride_frequency_hz"] = strides_result.get("stride_frequency", float("nan"))
        metrics["ground_contact_ms"] = strides_result.get("ground_contact_ms", float("nan"))
        metrics["strides"] = strides_result.get("strides", [])

        _prog(0.55, "Computing body angles...")
        metrics.update(analyze_angles(pose_df))

        _prog(0.68, "Computing joint angles...")
        metrics.update(calculate_angles(pose_df))

        _prog(0.78, "Computing velocity...")
        metrics.update(calculate_velocity(pose_df))

        _prog(0.87, "Computing arm swing...")
        arm = calculate_arm_swing(pose_df)
        metrics.update(arm)
        la = float(arm.get("left_arm_swing_amplitude", float("nan")) or float("nan"))
        ra = float(arm.get("right_arm_swing_amplitude", float("nan")) or float("nan"))
        if not (math.isnan(la) or math.isnan(ra)) and (la + ra) > 0:
            metrics["arm_swing_symmetry"] = abs(la - ra) / ((la + ra) / 2.0)

        _prog(0.93, "Estimating 40-yard time...")
        t = _extract_40_time(pose_df)
        metrics["extracted_40_time"] = t if t is not None else float("nan")

        sl = float(metrics["stride_length_m"] or float("nan"))
        if not math.isnan(sl) and (sl < 0.005 or sl > 0.5):
            metrics["status"] = "flagged"

    except Exception as exc:  # noqa: BLE001
        metrics["status"] = "error"
        metrics["error"] = str(exc)
        pose_df = pd.DataFrame()

    return metrics, pose_df


# ─── Skeleton overlay ─────────────────────────────────────────────────────────

def _create_skeleton_video(src_path: str, pose_df: pd.DataFrame, dst_path: str) -> None:
    """Render skeleton overlay on every frame and write to dst_path."""
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rows = pose_df[pose_df["frame_index"] == frame_idx]
        if not rows.empty:
            row = rows.iloc[0]

            def _pt(name: str) -> tuple[int, int] | None:
                vis = float(row.get(f"{name}_visibility", 0.0) or 0.0)
                if vis < _VIS_THRESHOLD:
                    return None
                px = int(float(row.get(f"{name}_x", 0.0) or 0.0) * w)
                py = int(float(row.get(f"{name}_y", 0.0) or 0.0) * h)
                return (px, py)

            for a_name, b_name in _SKELETON:
                pa, pb = _pt(a_name), _pt(b_name)
                if pa and pb:
                    color = _LEFT_COLOR if a_name.startswith("left") else _RIGHT_COLOR
                    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

            for name in _JOINTS:
                pt = _pt(name)
                if pt:
                    color = _LEFT_COLOR if name.startswith("left") else _RIGHT_COLOR
                    cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# ─── Grading helpers ──────────────────────────────────────────────────────────

def _grade(key: str, value: float) -> str:
    """Return 'green', 'yellow', 'red', or 'gray' for a metric value."""
    if key not in _BENCHMARKS or math.isnan(value):
        return "gray"
    bm = _BENCHMARKS[key]
    lo_elite, hi_elite = bm["elite"]
    lo_good, hi_good = bm["good"]

    if bm["lower_is_better"]:
        if value <= hi_elite:
            return "green"
        if value <= hi_good:
            return "yellow"
        return "red"
    else:
        if value >= lo_elite:
            return "green"
        if value >= lo_good:
            return "yellow"
        return "red"


def _metric_card_html(key: str, value: float, bm: dict) -> str:
    grade = _grade(key, value)
    color = _GRADE_HEX[grade]
    badge = _GRADE_LABEL[grade]
    val_str = f"{value:.2f}" if not math.isnan(value) else "—"
    return (
        f'<div style="background:#fff;border-radius:10px;padding:16px 20px;'
        f"border-left:5px solid {color};box-shadow:0 2px 6px rgba(0,0,0,.08);"
        f'margin-bottom:12px;">'
        f'<div style="font-size:.78em;color:#888;text-transform:uppercase;'
        f'letter-spacing:.06em;">{bm["label"]}</div>'
        f'<div style="display:flex;align-items:baseline;gap:6px;margin:4px 0;">'
        f'<span style="font-size:2em;font-weight:700;color:{color};">{val_str}</span>'
        f'<span style="font-size:.82em;color:#aaa;">{bm["unit"]}</span>'
        f'<span style="margin-left:auto;background:{color};color:#fff;border-radius:4px;'
        f'padding:2px 8px;font-size:.72em;white-space:nowrap;">{badge}</span>'
        f"</div>"
        f'<div style="font-size:.72em;color:#bbb;margin-top:2px;">{bm.get("tip", "")}</div>'
        f"</div>"
    )


# ─── Chart helpers ────────────────────────────────────────────────────────────

def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        st.info("Install matplotlib for charts: `pip install matplotlib`")
        return None


def _velocity_chart(pose_df: pd.DataFrame) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    hip_x = (pose_df["left_hip_x"] + pose_df["right_hip_x"]) / 2.0
    ts = pose_df["timestamp_sec"].values
    dt = np.diff(ts)
    vel = np.abs(np.diff(hip_x.values)) / (dt + 1e-9)
    vel_smooth = pd.Series(vel).rolling(7, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(ts[1:], vel_smooth, color="#1f77b4", linewidth=2)
    ax.fill_between(ts[1:], vel_smooth, alpha=0.12, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (norm/s)")
    ax.set_title("Velocity Over Time")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _stride_length_chart(strides: list[dict]) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    if not strides:
        st.info("No strides detected in this clip.")
        return

    left = [s for s in strides if s.get("foot") == "left"]
    right = [s for s in strides if s.get("foot") == "right"]

    fig, ax = plt.subplots(figsize=(9, 3))
    if left:
        ax.plot(
            [s["stride_index"] for s in left],
            [s["stride_length_m"] for s in left],
            "o-", color="#2ca02c", label="Left foot", linewidth=2, markersize=6,
        )
    if right:
        ax.plot(
            [s["stride_index"] for s in right],
            [s["stride_length_m"] for s in right],
            "s-", color="#ff7f0e", label="Right foot", linewidth=2, markersize=6,
        )
    ax.set_xlabel("Stride Index")
    ax.set_ylabel("Stride Length (norm)")
    ax.set_title("Stride Length Over Time")
    if left or right:
        ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _body_angle_chart(pose_df: pd.DataFrame) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    lhip_y = pose_df["left_hip_y"].ffill().bfill().values
    rhip_y = pose_df["right_hip_y"].ffill().bfill().values
    lsho_x = pose_df["left_shoulder_x"].ffill().bfill().values
    rsho_x = pose_df["right_shoulder_x"].ffill().bfill().values
    lsho_y = pose_df["left_shoulder_y"].ffill().bfill().values
    rsho_y = pose_df["right_shoulder_y"].ffill().bfill().values
    lhip_x = pose_df["left_hip_x"].ffill().bfill().values
    rhip_x = pose_df["right_hip_x"].ffill().bfill().values

    hip_cx = (lhip_x + rhip_x) / 2.0
    hip_cy = (lhip_y + rhip_y) / 2.0
    sho_cx = (lsho_x + rsho_x) / 2.0
    sho_cy = (lsho_y + rsho_y) / 2.0

    dx_abs = np.abs(sho_cx - hip_cx)
    dy_up = hip_cy - sho_cy  # positive when shoulder is above hip
    vert_angles = np.degrees(np.arctan2(dx_abs, dy_up))

    ts = pose_df["timestamp_sec"].values
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(ts, vert_angles, color="#d62728", linewidth=2)
    ax.axhline(20, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Transition threshold (20°)")
    ax.axhline(15, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Upright threshold (15°)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward Lean (°)")
    ax.set_title("Body Angle (Forward Lean) Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _elite_comparison_chart(metrics: dict) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    # 40-time gauge
    t = float(metrics.get("extracted_40_time") or float("nan"))
    if not math.isnan(t):
        fig, ax = plt.subplots(figsize=(9, 1.8))
        bar_color = _GRADE_HEX[_grade("extracted_40_time_raw", t)]
        # Manual grade for 40 time (not in _BENCHMARKS, graded separately)
        if t <= 4.5:
            bar_color = _GRADE_HEX["green"]
        elif t <= 4.8:
            bar_color = _GRADE_HEX["yellow"]
        else:
            bar_color = _GRADE_HEX["red"]

        ax.barh(["You"], [t], color=bar_color, height=0.5, zorder=3)
        ax.axvline(4.3, color="#28a745", linestyle="--", linewidth=1.5, label="Elite (4.3 s)")
        ax.axvline(4.5, color="#e6a817", linestyle="--", linewidth=1.5, label="Good (4.5 s)")
        ax.axvline(4.8, color="#dc3545", linestyle="--", linewidth=1.5, label="Average (4.8 s)")
        ax.set_xlim(3.8, 6.2)
        ax.set_xlabel("40-Yard Dash Time (s)")
        ax.set_title("Your Time vs NFL Combine Benchmarks")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Grade bar chart for all metrics
    keys = [
        "stride_length_m", "stride_frequency_hz", "ground_contact_ms",
        "drive_phase_angle", "hip_extension", "arm_swing_amplitude",
        "forward_lean_angle", "peak_velocity_norm",
    ]
    grade_scores, labels, colors = [], [], []
    for k in keys:
        v = float(metrics.get(k) or float("nan"))
        g = _grade(k, v)
        grade_scores.append({"green": 3, "yellow": 2, "red": 1, "gray": 0}[g])
        labels.append(_BENCHMARKS.get(k, {}).get("label", k))
        colors.append(_GRADE_HEX[g])

    if any(s > 0 for s in grade_scores):
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.bar(range(len(labels)), grade_scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Needs work", "Good", "Elite"])
        ax.set_title("Metric Grades vs Elite Benchmarks")
        ax.set_ylim(0, 3.6)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─── UI components ────────────────────────────────────────────────────────────

def _show_data_quality(metrics: dict) -> None:
    conf = metrics.get("avg_confidence", {})
    if not conf:
        return
    avg = sum(conf.values()) / len(conf)
    if avg >= 0.70:
        st.success(f"Tracking quality: **{avg:.0%}** — Good")
    elif avg >= 0.40:
        st.warning(f"Tracking quality: **{avg:.0%}** — Fair (some joints may be occluded)")
    else:
        st.error(f"Tracking quality: **{avg:.0%}** — Poor (results may be unreliable)")

    with st.expander("Per-joint tracking confidence"):
        cols = st.columns(3)
        for i, (joint, c) in enumerate(sorted(conf.items())):
            icon = "🟢" if c >= 0.70 else ("🟡" if c >= 0.40 else "🔴")
            cols[i % 3].markdown(f"{icon} **{joint}** {c:.0%}")


def _show_40_time_hero(t: float) -> None:
    if math.isnan(t):
        st.info("Could not estimate 40-yard dash time from this clip. "
                "Ensure the full run is captured with a stable camera angle.")
        return

    if t <= 4.3:
        label, color = "World-class", "#28a745"
    elif t <= 4.5:
        label, color = "NFL Combine elite", "#28a745"
    elif t <= 4.8:
        label, color = "Good — competitive level", "#e6a817"
    elif t <= 5.2:
        label, color = "Average — recreational", "#e6a817"
    else:
        label, color = "Below average", "#dc3545"

    st.markdown(
        f'<div style="text-align:center;background:#fff;border-radius:14px;padding:28px;'
        f"box-shadow:0 2px 10px rgba(0,0,0,.1);margin-bottom:18px;\">"
        f'<div style="font-size:.85em;color:#999;text-transform:uppercase;'
        f'letter-spacing:.12em;margin-bottom:6px;">Estimated 40-Yard Dash Time</div>'
        f'<div style="font-size:4.5em;font-weight:800;color:{color};line-height:1.05;">'
        f"{t:.2f}<span style='font-size:.4em;'>s</span></div>"
        f'<div style="font-size:1.05em;color:{color};margin-top:6px;">{label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _show_metrics_dashboard(metrics: dict) -> None:
    ordered_keys = [
        "stride_length_m",
        "stride_frequency_hz",
        "ground_contact_ms",
        "drive_phase_angle",
        "hip_extension",
        "arm_swing_amplitude",
        "forward_lean_angle",
        "transition_point_yards",
        "peak_velocity_norm",
    ]
    cols = st.columns(3)
    for i, key in enumerate(ordered_keys):
        if key not in _BENCHMARKS:
            continue
        value = float(metrics.get(key) or float("nan"))
        html = _metric_card_html(key, value, _BENCHMARKS[key])
        cols[i % 3].markdown(html, unsafe_allow_html=True)

    with st.expander("Additional metrics"):
        extra_items = [
            ("Arm Swing Symmetry", metrics.get("arm_swing_symmetry"), "0 = perfect"),
            ("Left Knee Angle", metrics.get("left_knee_angle_deg"), "°"),
            ("Right Knee Angle", metrics.get("right_knee_angle_deg"), "°"),
            ("Left Hip Angle", metrics.get("left_hip_angle_deg"), "°"),
            ("Right Hip Angle", metrics.get("right_hip_angle_deg"), "°"),
            ("Avg Velocity", metrics.get("avg_velocity_norm"), "norm/s"),
            ("Left Arm Swing", metrics.get("left_arm_swing_amplitude"), "norm"),
            ("Right Arm Swing", metrics.get("right_arm_swing_amplitude"), "norm"),
        ]
        cols2 = st.columns(4)
        for j, (label, val, unit) in enumerate(extra_items):
            v = float(val or float("nan"))
            val_str = f"{v:.3f} {unit}" if not math.isnan(v) else "—"
            cols2[j % 4].metric(label, val_str)


def _show_results(
    metrics: dict,
    pose_df: pd.DataFrame,
    video_bytes: bytes,
    name: str,
) -> None:
    st.subheader(f"Results — {name}")

    if metrics.get("status") == "flagged":
        st.warning(
            "Stride data was flagged as potentially implausible. "
            "Verify the video quality and camera angle."
        )

    _show_data_quality(metrics)
    _show_40_time_hero(float(metrics.get("extracted_40_time") or float("nan")))

    tab_vid, tab_dash, tab_charts, tab_compare = st.tabs(
        ["Video & Skeleton", "Metrics Dashboard", "Charts", "Elite Comparison"]
    )

    with tab_vid:
        st.subheader("Original Video")
        st.video(video_bytes)

        if not pose_df.empty:
            if st.button("Generate Skeleton Overlay Video", key="gen_skel"):
                st.session_state.pop("skeleton_bytes", None)
                with st.spinner("Rendering skeleton overlay — this may take a moment..."):
                    suf = Path(name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as src_tmp:
                        src_tmp.write(video_bytes)
                        src_path = src_tmp.name
                    dst_path = src_path + "_skel.mp4"
                    try:
                        _create_skeleton_video(src_path, pose_df, dst_path)
                        with open(dst_path, "rb") as fh:
                            st.session_state["skeleton_bytes"] = fh.read()
                    except Exception as exc:
                        st.error(f"Skeleton rendering failed: {exc}")
                    finally:
                        Path(src_path).unlink(missing_ok=True)
                        Path(dst_path).unlink(missing_ok=True)

            if "skeleton_bytes" in st.session_state:
                st.subheader("Skeleton Overlay")
                st.video(st.session_state["skeleton_bytes"])

    with tab_dash:
        _show_metrics_dashboard(metrics)

    with tab_charts:
        if not pose_df.empty:
            st.subheader("Velocity Over Time")
            _velocity_chart(pose_df)

            st.subheader("Stride Length Over Time")
            _stride_length_chart(metrics.get("strides", []))

            st.subheader("Body Angle Over Time")
            _body_angle_chart(pose_df)
        else:
            st.info("No pose data available for charts.")

    with tab_compare:
        st.markdown("""
**NFL Combine Reference Benchmarks**
- Elite 40 time: **4.3–4.5 s**
- Good stride length (real-world): **2.2–2.5 m**
- Elite peak velocity: **21–23 mph**

*Note: stride length and velocity are shown in normalised video-coordinate units
because pixel-to-metre calibration requires a known reference distance in frame.*
""")
        _elite_comparison_chart(metrics)

    # Download CSV for this video
    st.divider()
    _exclude = {"strides", "avg_confidence", "error"}
    csv_row = {k: v for k, v in metrics.items() if k not in _exclude}
    csv_bytes = pd.DataFrame([csv_row]).to_csv(index=False).encode()
    st.download_button(
        "⬇ Download results as CSV",
        data=csv_bytes,
        file_name=f"{Path(name).stem}_analysis.csv",
        mime="text/csv",
    )


# ─── Page: single video ───────────────────────────────────────────────────────

def _page_single() -> None:
    st.header("Single Video Analysis")
    st.markdown(
        "Upload a 40-yard dash video to extract pose keypoints and compute "
        "performance metrics with elite benchmarks."
    )

    uploaded = st.file_uploader(
        "Drop your video here or click to browse",
        type=_ACCEPTED_TYPES,
        key="single_upload",
        help="Supported: MP4, MOV, AVI",
    )

    if uploaded is None:
        for k in ("single_cache_key", "single_metrics", "single_pose_df",
                  "single_video_bytes", "single_video_name", "skeleton_bytes"):
            st.session_state.pop(k, None)
        st.info("Upload a video to get started.")
        return

    cache_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("single_cache_key") != cache_key:
        st.session_state["single_cache_key"] = cache_key
        st.session_state.pop("skeleton_bytes", None)

        video_bytes = uploaded.getvalue()
        suf = Path(uploaded.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        progress_bar = st.progress(0.0)
        status_txt = st.empty()

        def _on_progress(p: float, msg: str) -> None:
            progress_bar.progress(p)
            status_txt.text(msg)

        try:
            metrics, pose_df = _analyze_video(tmp_path, progress_cb=_on_progress)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        progress_bar.progress(1.0)
        status_txt.text("Analysis complete!")

        st.session_state["single_metrics"] = metrics
        st.session_state["single_pose_df"] = pose_df
        st.session_state["single_video_bytes"] = video_bytes
        st.session_state["single_video_name"] = uploaded.name

    metrics = st.session_state.get("single_metrics")
    pose_df = st.session_state.get("single_pose_df", pd.DataFrame())
    video_bytes = st.session_state.get("single_video_bytes", b"")
    video_name = st.session_state.get("single_video_name", uploaded.name)

    if metrics is None:
        return

    status = metrics.get("status")
    if status == "error":
        st.error(f"Analysis failed: {metrics.get('error')}")
        return
    if status == "no_pose":
        st.warning(
            "No human pose detected in this video. "
            "Ensure the athlete is fully visible and the video is clear."
        )
        return

    _show_results(metrics, pose_df, video_bytes, video_name)


# ─── Page: batch processing ───────────────────────────────────────────────────

def _page_batch() -> None:
    st.header("Batch Processing")
    st.markdown(
        "Upload multiple videos to process them all in sequence and download "
        "a combined CSV dataset."
    )

    uploaded_files = st.file_uploader(
        "Upload videos (select multiple)",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="batch_upload",
        help="Hold Ctrl / Cmd to select multiple files",
    )

    if not uploaded_files:
        st.info("Upload one or more videos to begin.")
        st.session_state.pop("batch_results", None)
        return

    st.markdown(f"**{len(uploaded_files)} video(s) selected**")

    if st.button("Process All Videos", type="primary"):
        results: list[dict] = []
        n = len(uploaded_files)
        progress_bar = st.progress(0.0)
        status_txt = st.empty()

        for idx, f in enumerate(uploaded_files):
            status_txt.text(f"Processing {f.name} ({idx + 1} / {n})...")
            suf = Path(f.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name

            try:
                metrics, _ = _analyze_video(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            metrics["video"] = f.name
            _exclude = {"strides", "avg_confidence", "error"}
            results.append({k: v for k, v in metrics.items() if k not in _exclude})
            progress_bar.progress((idx + 1) / n)

        status_txt.text(f"Done — processed {n} video(s).")
        st.session_state["batch_results"] = results

    if "batch_results" not in st.session_state:
        return

    results = st.session_state["batch_results"]
    df = pd.DataFrame(results)

    st.subheader(f"Batch Results — {len(df)} video(s)")

    # Status summary pills
    status_counts = df["status"].value_counts()
    pill_cols = st.columns(min(len(status_counts), 4))
    _status_colors = {"ok": "green", "flagged": "orange", "no_pose": "blue", "error": "red"}
    for i, (s, cnt) in enumerate(status_counts.items()):
        c = _status_colors.get(s, "gray")
        pill_cols[i].markdown(f"**:{c}[{s.upper()}]** — {cnt}")

    # Results table
    display_cols = [
        "video", "status", "extracted_40_time",
        "stride_length_m", "stride_frequency_hz", "ground_contact_ms",
        "peak_velocity_norm", "arm_swing_amplitude",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    _row_bg = {
        "ok": "#f0fff0", "flagged": "#fffbe6",
        "no_pose": "#e8f4f8", "error": "#fff0f0",
    }

    def _style_status(row):
        bg = _row_bg.get(str(row.get("status", "")), "")
        return [f"background-color: {bg}"] * len(row)

    numeric_cols = {c: "{:.4f}" for c in display_cols if c not in ("video", "status")}
    styled = (
        df[display_cols]
        .style.apply(_style_status, axis=1)
        .format(numeric_cols, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)

    # Download
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download full CSV",
        data=csv_bytes,
        file_name="batch_analysis.csv",
        mime="text/csv",
    )


# ─── App entry point ──────────────────────────────────────────────────────────

def run(port: int = 8200) -> None:
    """Launch the Streamlit app as a subprocess."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            __file__,
            "--server.port", str(port),
            "--server.headless", "true",
        ],
        check=True,
    )


def main() -> None:
    """Streamlit entry point — executed when ``streamlit run web_app.py`` is called."""
    st.set_page_config(
        page_title="Athlete Vision — 40-Yard Dash",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Athlete Vision")
        st.caption("40-Yard Dash Analysis")
        page = st.radio(
            "Navigation",
            ["Single Video Analysis", "Batch Processing"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("""
**Grade legend**
🟢 Elite  &nbsp; 🟡 Good  &nbsp; 🔴 Needs work
""")
        st.divider()
        st.markdown("""
**NFL Combine Benchmarks**
• 40 time: 4.3–4.5 s
• Stride length: 2.2–2.5 m
• Peak velocity: 21–23 mph
""")

    if page == "Single Video Analysis":
        _page_single()
    else:
        _page_batch()


if __name__ == "__main__":
    main()
