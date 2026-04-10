"""Camera calibration: pixel-to-metre conversion from video or user-provided reference."""

from __future__ import annotations

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

import cv2
import numpy as np

# Unit conversion factors → metres
_UNIT_TO_METRES: dict[str, float] = {
    "metres": 1.0,
    "meters": 1.0,
    "m": 1.0,
    "yards": 0.9144,
    "yard": 0.9144,
    "yd": 0.9144,
    "feet": 0.3048,
    "foot": 0.3048,
    "ft": 0.3048,
}

# 10 yards in metres (standard American-football yard-line spacing)
_TEN_YARDS_M: float = 10.0 * 0.9144  # 9.144 m


def unit_to_metres(value: float, unit: str) -> float:
    """Convert *value* from *unit* to metres.

    Parameters
    ----------
    value:
        Numeric measurement.
    unit:
        One of ``'metres'``/``'meters'``/``'m'``, ``'yards'``/``'yard'``/``'yd'``,
        ``'feet'``/``'foot'``/``'ft'``.

    Returns
    -------
    float
        Equivalent distance in metres.

    Raises
    ------
    ValueError
        When *unit* is not recognised.
    """
    key = unit.lower().strip()
    if key not in _UNIT_TO_METRES:
        supported = sorted(set(_UNIT_TO_METRES))
        raise ValueError(f"Unknown unit {unit!r}. Supported: {supported}")
    return value * _UNIT_TO_METRES[key]


def calibration_factor_from_reference(
    known_distance: float,
    unit: str,
    frame_span: float,
) -> float:
    """Compute the calibration factor from a user-supplied reference measurement.

    The calibration factor converts one normalised x-coordinate unit (as used by
    MediaPipe, where the entire frame width = 1.0) into metres.

    Parameters
    ----------
    known_distance:
        Real-world distance between two reference points visible in the frame.
    unit:
        Unit of *known_distance*: ``'yards'``, ``'meters'``/``'metres'``,
        or ``'feet'``.
    frame_span:
        Horizontal distance between the two reference points in normalised frame
        coordinates (0.0–1.0).  For example, if the points are 90 % of the
        frame width apart, pass ``0.9``.

    Returns
    -------
    float
        Calibration factor: metres per one normalised x-coordinate unit.

    Raises
    ------
    ValueError
        When *frame_span* is zero or negative, or *unit* is unrecognised.
    """
    if frame_span <= 0.0:
        raise ValueError(f"frame_span must be positive, got {frame_span}")
    metres = unit_to_metres(known_distance, unit)
    return metres / frame_span


def detect_yard_lines(
    video_path: str | Path,
    sample_frames: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length_ratio: float = 0.15,
    max_line_gap: int = 20,
    min_lines: int = 2,
    cluster_tolerance: float = 0.05,
) -> float | None:
    """Attempt to auto-detect yard line markings and return a calibration factor.

    Uses OpenCV Probabilistic Hough Transform to find near-vertical white stripe
    pairs characteristic of yard lines painted on an American-football field.
    When two or more distinct lines are found, the mean inter-line spacing is
    assumed to correspond to 10 yards (9.144 m), giving a calibration factor.

    This is a best-effort heuristic.  It works best on broadcast footage where
    10-yard hash marks are clearly visible in a landscape frame.  Returns
    ``None`` when detection is not confident enough.

    Parameters
    ----------
    video_path:
        Path to the video file.
    sample_frames:
        Number of evenly-spaced frames to sample.
    canny_low, canny_high:
        Hysteresis thresholds for Canny edge detection.
    hough_threshold:
        Accumulator threshold for the Hough transform (higher = fewer, stronger lines).
    min_line_length_ratio:
        Minimum line length as a fraction of the frame width.
    max_line_gap:
        Maximum allowed gap (pixels) inside a line segment.
    min_lines:
        Minimum number of distinct line clusters required to return a result.
    cluster_tolerance:
        Two line positions closer than this fraction of the frame width are
        merged into one cluster.

    Returns
    -------
    float | None
        Estimated calibration factor (metres per normalised x-unit), or ``None``
        when detection failed or confidence was insufficient.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or frame_width <= 0 or frame_height <= 0:
            return None

        min_line_length = max(1, int(frame_width * min_line_length_ratio))
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

        all_x_positions: list[float] = []

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue

            # Focus on the lower half of the frame where field markings are clearest
            h = frame.shape[0]
            roi = frame[h // 2 :, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, canny_low, canny_high)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
            )

            if lines is None:
                continue

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy == 0:
                    continue
                # Keep near-vertical lines (angle < 20° from vertical)
                angle = math.degrees(math.atan2(dx, dy))
                if angle < 20.0:
                    x_mid = (x1 + x2) / 2.0 / frame_width
                    all_x_positions.append(x_mid)

        if len(all_x_positions) < min_lines:
            logger.warning(
                "detect_yard_lines: insufficient near-vertical lines found in %s "
                "(found %d, need %d); falling back to scale factor 1.0",
                video_path,
                len(all_x_positions),
                min_lines,
            )
            return None

        # Cluster nearby x-positions into distinct yard lines
        x_arr = np.array(sorted(all_x_positions))
        clusters: list[list[float]] = []
        current: list[float] = [float(x_arr[0])]

        for x in x_arr[1:]:
            if float(x) - current[-1] < cluster_tolerance:
                current.append(float(x))
            else:
                clusters.append(current)
                current = [float(x)]
        clusters.append(current)

        if len(clusters) < min_lines:
            logger.warning(
                "detect_yard_lines: too few distinct yard-line clusters in %s "
                "(found %d, need %d); falling back to scale factor 1.0",
                video_path,
                len(clusters),
                min_lines,
            )
            return None

        cluster_centres = [float(np.mean(c)) for c in clusters]
        spacings = np.diff(cluster_centres)
        mean_spacing = float(np.mean(spacings))

        if mean_spacing <= 0:
            logger.warning(
                "detect_yard_lines: computed non-positive inter-line spacing for %s; "
                "falling back to scale factor 1.0",
                video_path,
            )
            return None

        # Assume inter-line spacing = 10 yards = 9.144 m
        return _TEN_YARDS_M / mean_spacing

    finally:
        cap.release()


def calibrate(
    video_path: str | Path | None = None,
    *,
    calibration_distance: float | None = None,
    calibration_unit: str = "yards",
    frame_span: float | None = None,
    auto_detect: bool = True,
) -> float:
    """Return a calibration factor (metres per normalised x-unit).

    Resolution order (first match wins):

    1. User-provided *calibration_distance* + *frame_span* — explicit reference.
    2. Auto-detection from *video_path* via Hough-line yard-line detection.
    3. Default fallback of ``1.0`` (normalised units, no real-world scale).

    Parameters
    ----------
    video_path:
        Path to the video file.  Required for auto-detection.
    calibration_distance:
        Known real-world distance (e.g., ``40`` for a 40-yard reference line).
    calibration_unit:
        Unit for *calibration_distance*: ``'yards'``, ``'meters'``/``'metres'``,
        or ``'feet'``.  Default ``'yards'``.
    frame_span:
        Normalised frame width (0–1) spanned by the reference distance.
        When ``None`` and *calibration_distance* is given, defaults to ``0.9``
        (reference markers assumed to span 90 % of the frame width).
    auto_detect:
        Whether to attempt automatic yard-line detection when no explicit
        reference is provided.  Default ``True``.

    Returns
    -------
    float
        Calibration factor: metres per one normalised x-coordinate unit.
        Returns ``1.0`` when no calibration source is available.
    """
    # 1. Explicit user reference
    if calibration_distance is not None:
        span = frame_span if frame_span is not None else 0.9
        return calibration_factor_from_reference(calibration_distance, calibration_unit, span)

    # 2. Auto-detection from video
    if auto_detect and video_path is not None:
        factor = detect_yard_lines(video_path)
        if factor is not None:
            return factor

    # 3. Default — normalised units unchanged
    return 1.0
