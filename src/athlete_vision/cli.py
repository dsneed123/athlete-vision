"""CLI entry point for Athlete Vision."""

import json
import sys
from pathlib import Path

import click
import pandas as pd

from .batch_processor import batch_process, generate_html_report, print_summary
from .pipeline import print_pipeline_summary, run_pipeline
from .pose_estimator import PoseEstimator
from .video_downloader import SEARCH_QUERIES, search_and_download

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
DOWNLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


@click.group()
def main() -> None:
    """Athlete Vision — pose estimation and movement analysis from video."""


@main.command("process")
@click.option(
    "--video-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True),
    help="Directory containing video files (.mp4, .mov, .avi).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(writable=True),
    help="Output directory where per-video CSV files are written.",
)
@click.option(
    "--model-complexity",
    default=1,
    show_default=True,
    type=click.IntRange(0, 2),
    help="MediaPipe model complexity (0=lite, 1=full, 2=heavy).",
)
def process_cmd(video_dir: str, output: str, model_complexity: int) -> None:
    """Process athlete videos and extract pose keypoint data."""
    video_dir_path = Path(video_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in video_dir_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        click.echo(f"No video files found in {video_dir_path}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(videos)} video(s) in {video_dir_path}")

    with PoseEstimator(model_complexity=model_complexity) as estimator:
        for video in videos:
            click.echo(f"Processing: {video.name}")
            try:
                df = estimator.process_video(str(video))
            except ValueError as exc:
                click.echo(f"  ERROR: {exc}", err=True)
                continue

            out_file = output_path / f"{video.stem}_keypoints.csv"
            df.to_csv(out_file, index=False)
            click.echo(f"  Saved {len(df)} frames -> {out_file}")

            if "avg_confidence" in df.attrs:
                for joint, conf in df.attrs["avg_confidence"].items():
                    click.echo(f"    {joint}: {conf:.3f}")


@main.command("download")
@click.option(
    "--count",
    default=40,
    show_default=True,
    type=click.IntRange(1, 200),
    help="Number of videos to download.",
)
@click.option(
    "--output-dir",
    default="./videos",
    show_default=True,
    type=click.Path(writable=True),
    help="Directory to save downloaded videos and metadata.",
)
@click.option(
    "--model-complexity",
    default=1,
    show_default=True,
    type=click.IntRange(0, 2),
    help="MediaPipe model complexity for pose extraction (0=lite, 1=full, 2=heavy).",
)
def download_cmd(count: int, output_dir: str, model_complexity: int) -> None:
    """Download 40-yard dash videos from YouTube and run pose extraction."""
    output_path = Path(output_dir)

    # --- 1. Download videos ---
    click.echo(f"Downloading up to {count} videos to {output_path} ...")
    downloaded, failed_downloads = search_and_download(
        queries=SEARCH_QUERIES,
        count=count,
        output_dir=output_path,
    )

    if not downloaded:
        click.echo("No videos downloaded.", err=True)
        sys.exit(1)

    click.echo(f"\nDownloaded {len(downloaded)} video(s). Running pose extraction ...")

    # --- 2. Run pose extraction on all videos in output directory ---
    videos = sorted(
        p
        for p in output_path.iterdir()
        if p.suffix.lower() in DOWNLOAD_EXTENSIONS
    )

    metadata_path = output_path / "metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}

    summary_rows: list[dict] = []

    with PoseEstimator(model_complexity=model_complexity) as estimator:
        for video in videos:
            click.echo(f"Processing: {video.name}")
            try:
                df = estimator.process_video(str(video))
            except ValueError as exc:
                click.echo(f"  ERROR: {exc}", err=True)
                continue

            out_file = output_path / f"{video.stem}_keypoints.csv"
            df.to_csv(out_file, index=False)
            click.echo(f"  Saved {len(df)} frames -> {out_file}")

            # Match video to metadata by the video_id prefix in the filename
            video_id = video.stem.split("_")[0]
            meta = metadata.get(video_id, {})

            avg_conf: float | None = None
            if "avg_confidence" in df.attrs:
                confs = df.attrs["avg_confidence"]
                if confs:
                    avg_conf = sum(confs.values()) / len(confs)

            summary_rows.append(
                {
                    "video_id": video_id,
                    "title": meta.get("title", video.stem),
                    "athlete_name": meta.get("athlete_name"),
                    "known_time_sec": meta.get("known_time"),
                    "duration_sec": meta.get("duration"),
                    "frames_extracted": len(df),
                    "avg_pose_confidence": round(avg_conf, 4) if avg_conf is not None else None,
                    "keypoints_csv": str(out_file),
                }
            )

    if not summary_rows:
        click.echo("No videos were successfully processed.", err=True)
        sys.exit(1)

    # --- 3. Output summary CSV and comparison table ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    click.echo(f"\nDataset summary -> {summary_path}")

    with_times = summary_df[summary_df["known_time_sec"].notna()]
    if not with_times.empty:
        click.echo(
            f"\n{'Athlete / Title':<32} {'Known Time':>10} {'Frames':>8} {'Avg Conf':>10}"
        )
        click.echo("-" * 64)
        for _, row in with_times.iterrows():
            label = str(row["athlete_name"] or row["title"])[:31]
            conf = row["avg_pose_confidence"]
            click.echo(
                f"{label:<32} {row['known_time_sec']:>10.3f}"
                f" {int(row['frames_extracted']):>8}"
                f" {(conf if conf is not None else 0.0):>10.4f}"
            )
    else:
        click.echo("No videos with known 40-yard dash times found in titles.")

    if failed_downloads:
        click.echo(f"\n{len(failed_downloads)} download(s) failed:", err=True)
        for f in failed_downloads:
            click.echo(f"  {f['video_id']}  {f['title'][:60]}  — {f['error']}", err=True)


@main.command("batch")
@click.option(
    "--video-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True),
    help="Directory containing video files to process.",
)
@click.option(
    "--output",
    default="dataset.csv",
    show_default=True,
    type=click.Path(writable=True),
    help="Path for the output CSV dataset.",
)
@click.option(
    "--model-complexity",
    default=1,
    show_default=True,
    type=click.IntRange(0, 2),
    help="MediaPipe model complexity (0=lite, 1=full, 2=heavy).",
)
@click.option(
    "--min-frames",
    default=100,
    show_default=True,
    type=int,
    help="Minimum frame count; shorter clips are flagged.",
)
@click.option(
    "--max-tracking-loss",
    default=0.10,
    show_default=True,
    type=float,
    help="Maximum fraction of frames with a critical joint NaN before flagging.",
)
@click.option(
    "--min-confidence",
    default=0.85,
    show_default=True,
    type=float,
    help="Minimum average MediaPipe visibility for critical joints before flagging.",
)
def batch_cmd(
    video_dir: str,
    output: str,
    model_complexity: int,
    min_frames: int,
    max_tracking_loss: float,
    min_confidence: float,
) -> None:
    """Batch process videos and validate extracted 40-yard dash times.

    Runs pose estimation, stride analysis, angle calculations, velocity, and
    arm-swing analysis on every video in VIDEO_DIR.  Results are written to
    OUTPUT as a CSV dataset and a full summary is printed to stdout.
    Where metadata.json contains known 40 times, accuracy is reported.
    """
    video_dir_path = Path(video_dir)
    output_path = Path(output)

    click.echo(f"Processing videos in {video_dir_path} ...")

    try:
        df = batch_process(
            video_dir_path,
            output_path,
            model_complexity,
            min_frames=min_frames,
            max_tracking_loss_ratio=max_tracking_loss,
            min_avg_confidence=min_confidence,
        )
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"\nDataset written -> {output_path}  ({len(df)} row(s))")
    print_summary(df)


@main.command("web")
@click.option(
    "--port",
    default=8200,
    show_default=True,
    type=click.IntRange(1024, 65535),
    help="Port to serve the Streamlit web interface on.",
)
def web_cmd(port: int) -> None:
    """Launch the Streamlit web interface for video upload and analysis."""
    from .web_app import run

    click.echo(f"Starting Athlete Vision web interface on http://localhost:{port}")
    run(port)


@main.command("pipeline")
@click.option(
    "--video-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True),
    help="Directory containing video files (.mp4, .mov, .avi, .mkv).",
)
@click.option(
    "--output",
    default="pipeline_output.csv",
    show_default=True,
    type=click.Path(writable=True),
    help="Output CSV path.",
)
@click.option(
    "--athlete-id",
    default=None,
    help="Athlete identifier for all rows (default: video filename stem).",
)
@click.option(
    "--model-complexity",
    default=1,
    show_default=True,
    type=click.IntRange(0, 2),
    help="MediaPipe model complexity (0=lite, 1=full, 2=heavy).",
)
@click.option(
    "--calibration-distance",
    default=None,
    type=float,
    help=(
        "Known reference distance visible in the frame "
        "(e.g. 40 for a 40-yard span).  Used together with "
        "--calibration-unit to compute a pixel-to-metre scale factor."
    ),
)
@click.option(
    "--calibration-unit",
    default="yards",
    show_default=True,
    type=click.Choice(["yards", "meters", "feet"], case_sensitive=False),
    help="Unit for --calibration-distance.",
)
@click.option(
    "--min-frames",
    default=100,
    show_default=True,
    type=int,
    help="Minimum frame count; shorter clips are flagged REVIEW.",
)
@click.option(
    "--max-tracking-loss",
    default=0.10,
    show_default=True,
    type=float,
    help="Maximum fraction of frames with a critical joint NaN before REVIEW.",
)
@click.option(
    "--min-confidence",
    default=0.85,
    show_default=True,
    type=float,
    help="Minimum average MediaPipe visibility for critical joints before REVIEW.",
)
def pipeline_cmd(
    video_dir: str,
    output: str,
    athlete_id: str | None,
    model_complexity: int,
    calibration_distance: float | None,
    calibration_unit: str,
    min_frames: int,
    max_tracking_loss: float,
    min_confidence: float,
) -> None:
    """Run the full analysis pipeline on a directory of videos.

    Produces a CSV with one row per video containing: athlete_id,
    video_filename, forty_time, stride_length, stride_frequency,
    ground_contact_ms, drive_phase_angle, hip_extension,
    arm_swing_symmetry, forward_lean_angle, transition_point_yards,
    peak_velocity_mph, data_quality.

    Videos are flagged REVIEW when confidence is low, the clip is too
    short, tracking was lost too often, or the camera angle is non-standard.

    When --calibration-distance is provided, stride lengths and velocities
    are reported in real-world metres.  Without it, values are in normalised
    frame units (1 unit = full frame width).
    """
    from .calibration import calibrate

    video_dir_path = Path(video_dir)
    output_path = Path(output)

    cal_factor = calibrate(
        calibration_distance=calibration_distance,
        calibration_unit=calibration_unit,
    )
    if calibration_distance is not None:
        click.echo(
            f"Calibration: {calibration_distance} {calibration_unit} → "
            f"factor = {cal_factor:.4f} m/unit"
        )

    click.echo(f"Running pipeline on {video_dir_path} ...")

    try:
        df, stats = run_pipeline(
            video_dir_path,
            output_path,
            model_complexity=model_complexity,
            athlete_id=athlete_id,
            calibration_factor=cal_factor,
            min_frames=min_frames,
            max_tracking_loss_ratio=max_tracking_loss,
            min_avg_confidence=min_confidence,
        )
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"\nOutput written -> {output_path}  ({len(df)} row(s))")
    print_pipeline_summary(df, stats)


@main.command("report")
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Path to the batch CSV dataset produced by 'athlete-vision batch'.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(writable=True),
    help="Output HTML file path (default: <csv-name>.html).",
)
def report_cmd(csv_path: str, output: str | None) -> None:
    """Generate a visual HTML report with charts from a batch CSV dataset."""
    csv_p = Path(csv_path)
    out_p = Path(output) if output else csv_p.with_suffix(".html")

    df = pd.read_csv(csv_p)
    if df.empty:
        click.echo("CSV is empty — nothing to report.", err=True)
        sys.exit(1)

    generate_html_report(df, out_p)
    click.echo(f"Report written -> {out_p}")
