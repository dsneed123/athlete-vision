"""CLI entry point for Athlete Vision."""

import os
import sys
from pathlib import Path

import click
import pandas as pd

from .pose_estimator import PoseEstimator

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


@click.command()
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
def main(video_dir: str, output: str, model_complexity: int) -> None:
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
