#!/usr/bin/env python3
"""
Launcher for pose detection. Forwards CLI args to `pose_video.py`.
"""

import sys

from pose_video import main as run_pose


def main():
    # `pose_video.main()` uses argparse and reads from sys.argv by default,
    # so simply call it to forward whatever arguments were passed to this script.
    run_pose()


if __name__ == "__main__":
    main()
