"""CLI entry point for Athlete Vision."""

import json
import sys
from pathlib import Path

import click
import pandas as pd

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
    downloaded = search_and_download(
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
