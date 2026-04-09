# Athlete Vision

Pose estimation and movement analysis from athlete video using MediaPipe.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

```bash
athlete-vision --video-dir /path/to/videos --output /path/to/output
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--video-dir` | required | Directory containing `.mp4`, `.mov`, or `.avi` files |
| `--output` | required | Directory where per-video CSV files are written |
| `--model-complexity` | `1` | MediaPipe model complexity: `0` (lite), `1` (full), `2` (heavy) |

Each video produces a CSV named `<stem>_keypoints.csv` with columns:

- `frame_index`, `timestamp_sec`
- `{joint}_x`, `{joint}_y`, `{joint}_z`, `{joint}_visibility` for each tracked joint

Tracked joints: ankles, knees, hips, shoulders, elbows, wrists (left and right).

## Running tests

```bash
pip install pytest
pytest tests/
```
