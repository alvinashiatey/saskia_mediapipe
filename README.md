# Pose Draw

Pose Draw is a macOS desktop app that tracks wrist movement with MediaPipe Pose
and lets you draw on top of video playback.

Quick start (dev)

```bash
uv sync
uv run main.py
```

Build & distribute (macOS, unsigned)

1) Install build tooling:

```bash
uv add --dev pyinstaller
```

2) Build the app bundle:

```bash
uv run pyinstaller pose_draw.spec
```

This produces `dist/Pose Draw.app`.

3) Share the app:

- Zip it:
  ```bash
  ditto -c -k --sequesterRsrc --keepParent "dist/Pose Draw.app" "Pose Draw.zip"
  ```
- Or create a DMG with your preferred tool.

Gatekeeper note: because the app is unsigned, users will need to right-click
the app and choose Open the first time they run it.
