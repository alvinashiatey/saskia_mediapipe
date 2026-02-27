"""
MediaPipe helpers: model download, landmarker construction,
skeleton rendering, and wrist-pixel extraction.
"""

import os
import urllib.request
from typing import Optional

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from pose_draw.constants import (
    LM_LEFT_WRIST,
    LM_RIGHT_WRIST,
    MODEL_PATH,
    MODEL_URL,
    POSE_CONNECTIONS,
)


def download_model(path: str = MODEL_PATH) -> None:
    if not os.path.exists(path):
        print("Downloading pose landmarker model…")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Model ready.")


def build_pose_landmarker(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence:  float = 0.5,
) -> vision.PoseLandmarker:
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker: vision.PoseLandmarker, bgr_frame):
    """Run pose detection on a BGR numpy frame. Returns PoseLandmarkerResult."""
    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
    )
    return landmarker.detect(mp_image)


def draw_pose_on_frame(frame, pose_landmarks_list) -> None:
    """Draw the skeleton overlay onto a BGR frame in-place."""
    h, w = frame.shape[:2]
    for landmarks in pose_landmarks_list:
        for start_idx, end_idx in POSE_CONNECTIONS:
            s, e = landmarks[start_idx], landmarks[end_idx]
            cv2.line(
                frame,
                (int(s.x * w), int(s.y * h)),
                (int(e.x * w), int(e.y * h)),
                (0, 0, 200), 2,
            )
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                       3, (0, 200, 0), -1)


def get_wrist_px(
    landmarks,
    wrist_idx: int,
    w: int,
    h: int,
    min_visibility: float = 0.5,
) -> Optional[tuple[int, int]]:
    """
    Return pixel coordinates of a wrist landmark, or None if not reliably visible.
    """
    lm = landmarks[wrist_idx]
    if hasattr(lm, "visibility") and lm.visibility < min_visibility:
        return None
    return (int(lm.x * w), int(lm.y * h))
