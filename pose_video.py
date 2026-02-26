#!/usr/bin/env python3
"""
Simple MediaPipe Pose detection on a video file.
Saves annotated output to disk and optionally displays it.
"""

import argparse
import glob
import os
import time
import urllib.request

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# MediaPipe Pose landmark connections (from the official spec)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # face left
    (0, 4), (4, 5), (5, 6), (6, 8),        # face right
    (9, 10),                                 # mouth
    (11, 12),                                # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm
    (11, 23), (12, 24), (23, 24),            # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # right leg
]


def download_model(path: str = MODEL_PATH) -> None:
    if not os.path.exists(path):
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Done.")


def build_pose_landmarker(
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> vision.PoseLandmarker:
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_landmarks_on_frame(frame, pose_landmarks_list) -> None:
    """Draw pose landmarks onto the frame in-place using plain OpenCV."""
    h, w = frame.shape[:2]
    for landmarks in pose_landmarks_list:
        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x0, y0 = int(start.x * w), int(start.y * h)
            x1, y1 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # Draw landmark points
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)


def find_first_video_in_folder(folder: str = "video") -> str | None:
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        files = sorted(glob.glob(os.path.join(folder, ext)))
        if files:
            return files[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose on a video.")
    parser.add_argument(
        "--input", "-i",
        help="Path to input video. If omitted, uses first video in ./video/",
        default=None,
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to output video (mp4). Default: ./video/output.mp4",
        default="video/output.mp4",
    )
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Show annotated video in a window while processing",
    )
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or find_first_video_in_folder("video")
    if not input_path or not os.path.exists(input_path):
        print("No input video found. Place a video in 'video/' or pass --input.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    download_model()
    pose = build_pose_landmarker(
        args.min_detection_confidence, args.min_tracking_confidence
    )

    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            mp_image = Image(
                image_format=ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results = pose.detect(mp_image)

            if results.pose_landmarks:
                draw_landmarks_on_frame(frame, results.pose_landmarks)

            cur_time = time.time()
            elapsed = cur_time - prev_time
            fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: ?"
            prev_time = cur_time

            cv2.putText(
                frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )
            out.write(frame)

            if args.display:
                cv2.imshow("Pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user.")
                    break
    finally:
        cap.release()
        out.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Output saved to {args.output}")


if __name__ == "__main__":
    main()
