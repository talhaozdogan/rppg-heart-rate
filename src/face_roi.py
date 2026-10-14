"""Face tracking + skin-ROI extraction using MediaPipe FaceLandmarker.

Uses the Tasks API (mediapipe 1.x dropped mp.solutions.face_mesh); same 468-point
mesh. Each frame returns the mean R/G/B over the forehead and both cheeks, taken as
the convex hull of a curated landmark set per region. Eyes/mouth/hair are excluded
as noise. A `face_found` flag lets the pipeline mark missing frames rather than crash.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Face Mesh landmark indices inside each skin region; convex hulls form the ROIs.
FOREHEAD = [107, 66, 105, 104, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336]
LEFT_CHEEK = [205, 206, 207, 213, 192, 147, 123, 116, 117, 118, 101, 36, 203]
RIGHT_CHEEK = [425, 426, 427, 433, 416, 376, 352, 345, 346, 347, 330, 266, 423]
REGIONS = (FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK)


@dataclass
class FrameRGB:
    """Mean R,G,B over the skin ROI for a single frame."""
    t: float            # timestamp (seconds)
    r: float
    g: float
    b: float
    face_found: bool


class FaceROIExtractor:
    """Wraps a MediaPipe FaceLandmarker in VIDEO running mode."""

    def __init__(self, model_path: str = "models/face_landmarker.task") -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def roi_mask(self, landmarks, h: int, w: int) -> np.ndarray:
        """Build a binary mask (h, w) of the forehead + both cheeks."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in REGIONS:
            pts = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in region],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        return mask

    def extract(self, frame_bgr: np.ndarray, t_ms: int) -> FrameRGB:
        """Mean ROI RGB for one BGR frame. t_ms must be strictly increasing (ms)."""
        frame_rgb, _ = self.extract_detailed(frame_bgr, t_ms)
        return frame_rgb

    def extract_detailed(self, frame_bgr: np.ndarray, t_ms: int):
        """Like extract, but also returns the landmarks (or None) for the overlay."""
        t = t_ms / 1000.0
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, int(t_ms))

        if not result.face_landmarks:
            return FrameRGB(t, np.nan, np.nan, np.nan, False), None

        landmarks = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        mask = self.roi_mask(landmarks, h, w)
        pixels = rgb_frame[mask > 0]
        if pixels.size == 0:
            return FrameRGB(t, np.nan, np.nan, np.nan, False), landmarks

        r, g, b = pixels.mean(axis=0)
        return FrameRGB(t, float(r), float(g), float(b), True), landmarks

    def close(self) -> None:
        self._landmarker.close()
