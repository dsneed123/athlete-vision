"""MediaPipe Pose wrapper for per-frame keypoint extraction.

Uses the new mediapipe.tasks API (PoseLandmarker) which is required for
mediapipe >= 0.10.30 / Python 3.14+.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# MediaPipe landmark indices for the joints we care about
TRACKED_LANDMARKS = {
    "left_ankle": 27,
    "right_ankle": 28,
    "left_knee": 25,
    "right_knee": 26,
    "left_hip": 23,
    "right_hip": 24,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

# Default model path — look next to this file, then in the repo root
_MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "pose_landmarker.task",
    Path(__file__).resolve().parent.parent.parent / "pose_landmarker.task",
]


def _find_model() -> str:
    for p in _MODEL_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "pose_landmarker.task not found. Download it from "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = _find_model()
        self._options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(self._options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def process_frame(self, frame_bgr, timestamp_ms: int):
        """Process a single BGR frame and return landmarks or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def process_video(self, video_path: str) -> pd.DataFrame:
        """Process a video file and return a DataFrame of keypoint coordinates.

        Each row represents one frame. Columns follow the pattern:
            {joint}_{axis}  where axis in {x, y, z, visibility}
        Plus:
            frame_index, timestamp_sec

        Also computes average tracking confidence per joint and attaches it as
        metadata in the DataFrame's attrs dict under key "avg_confidence".
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rows = []

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_index * 1000.0 / fps)
            landmarks = self.process_frame(frame, timestamp_ms)

            row: dict = {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / fps,
            }

            if landmarks:
                for joint, idx in TRACKED_LANDMARKS.items():
                    p = landmarks[idx]
                    row[f"{joint}_x"] = p.x
                    row[f"{joint}_y"] = p.y
                    row[f"{joint}_z"] = p.z
                    row[f"{joint}_visibility"] = p.visibility
            else:
                for joint in TRACKED_LANDMARKS:
                    row[f"{joint}_x"] = np.nan
                    row[f"{joint}_y"] = np.nan
                    row[f"{joint}_z"] = np.nan
                    row[f"{joint}_visibility"] = np.nan

            rows.append(row)
            frame_index += 1

        cap.release()

        df = pd.DataFrame(rows)

        # Average tracking confidence per joint (ignoring NaN frames)
        avg_confidence = {}
        for joint in TRACKED_LANDMARKS:
            col = f"{joint}_visibility"
            if col in df.columns:
                avg_confidence[joint] = float(df[col].mean(skipna=True))
        df.attrs["avg_confidence"] = avg_confidence

        return df
