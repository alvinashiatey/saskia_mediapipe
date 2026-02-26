# Pose detection with MediaPipe

This repository includes a small script to run MediaPipe Pose on a video.

Files added:

- `pose_video.py` — reads a video from `video/` (or from `--input`), annotates pose landmarks, and writes `video/output.mp4` by default.
- `requirements.txt` — dependencies to install.

Quick start

1. Create and activate a virtual environment (if needed):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put your video file in the `video/` folder (e.g. `video/myclip.mp4`) or pass `--input`.

4. Run the pose detector:

```bash
python pose_video.py --input video/myclip.mp4 --output video/output.mp4 --display
```

Or (using first video found in `video/`):

```bash
python pose_video.py --display
```

Press `q` to stop the display window early. The annotated output will be saved to `video/output.mp4` by default.
