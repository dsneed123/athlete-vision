"""MediaPipe Pose wrapper for per-frame keypoint extraction."""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)

            row: dict = {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / fps,
            }

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                for joint, idx in TRACKED_LANDMARKS.items():
                    p = lm[idx]
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
