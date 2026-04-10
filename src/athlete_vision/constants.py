"""Shared analysis constants for the athlete_vision package."""

# Plausible 40-yard dash time range (seconds).
# Used for filtering implausible times in pipeline, batch_processor,
# velocity_analyzer, and video_downloader.
FORTY_TIME_MIN: float = 3.5
FORTY_TIME_MAX: float = 6.5

# Data-quality thresholds (pipeline quality checks)
CONFIDENCE_THRESHOLD: float = 0.85   # Average visibility for critical joints
MIN_FRAMES: int = 100                 # Minimum frame count for reliable analysis
MAX_TRACKING_LOSS: float = 0.10      # Max fraction of frames with any critical joint NaN

# Pose-plausibility threshold: fraction of frames failing any biomechanical
# check that triggers the IMPLAUSIBLE_POSE flag.
MAX_IMPLAUSIBLE_RATIO: float = 0.05

# Video aspect-ratio check: allowed relative deviation from standard ratios
# (16:9, 4:3, 3:2).
ASPECT_TOLERANCE: float = 0.15
