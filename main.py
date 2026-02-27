#!/usr/bin/env python3
"""
main.py — entrypoint for the Pose Draw application.

Usage:
    uv run main.py                         # open app, load video via UI
    uv run main.py -i path/to/video.mp4   # pre-load a video at startup
"""

import argparse

from pose_draw.app import PoseDrawApp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose Draw — wrist-tracking drawing tool")
    p.add_argument(
        "--input", "-i", default=None, metavar="VIDEO",
        help="Video file to pre-load at startup (optional).",
    )
    p.add_argument(
        "--min-detection-confidence", type=float, default=0.5, metavar="F",
        help="MediaPipe minimum pose detection confidence (default: 0.5).",
    )
    p.add_argument(
        "--min-tracking-confidence", type=float, default=0.5, metavar="F",
        help="MediaPipe minimum pose tracking confidence (default: 0.5).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = PoseDrawApp(
        initial_video=args.input,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    app.run()


if __name__ == "__main__":
    main()
