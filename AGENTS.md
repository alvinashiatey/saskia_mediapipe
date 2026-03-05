# AGENTS.md

This document provides coding standards and commands for AI agents working in this repository.

## Project Overview

**saskia-mediapipe** is a Python desktop GUI application for pose detection and wrist-tracking drawing using MediaPipe and CustomTkinter. It processes video files to track wrist movements and create drawings overlaid on video output.

- **Language:** Python 3.11+
- **Package Manager:** `uv` (modern Python package manager)
- **GUI Framework:** CustomTkinter (Tkinter-based)
- **ML Framework:** MediaPipe (pose detection)
- **Video Processing:** OpenCV

## Build/Run/Lint Commands

### Running the Application
```bash
# Launch GUI (load video via UI)
uv run main.py

# Pre-load a video at startup
uv run main.py -i path/to/video.mp4

# Using entry point script
uv run pose-draw

# With custom MediaPipe confidence thresholds
uv run main.py -i video.mp4 --min-detection-confidence 0.7 --min-tracking-confidence 0.6
```

### Dependency Management
```bash
# Install/sync dependencies from uv.lock
uv sync

# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv lock --upgrade

# Add build tooling
uv add --dev pyinstaller
```

### Linting and Formatting
```bash
# Check code style and lint issues
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Check and format in one go
ruff check --fix . && ruff format .
```

### Testing
**Note:** Currently no test framework is configured. If adding tests:
```bash
# Install pytest
uv add --dev pytest

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_filename.py

# Run a specific test function
uv run pytest tests/test_filename.py::test_function_name

# Run with verbose output
uv run pytest -v
```

## Code Style Guidelines

### Imports
- Use absolute imports from the `pose_draw` package
- Group imports in standard order: stdlib, third-party, local
- Example:
  ```python
  import os
  import queue
  import threading
  from typing import Optional, Callable
  
  import cv2
  import numpy as np
  from mediapipe import Image, ImageFormat
  
  from pose_draw.brush import BrushSettings, hex_to_bgr
  from pose_draw.constants import LM_LEFT_WRIST, LM_RIGHT_WRIST
  ```

### Type Hints
- **Always use type hints** for function parameters and return types
- Use modern Python 3.11+ syntax: `tuple[int, int]` not `Tuple[int, int]`
- Use `Optional[T]` for nullable types
- Example:
  ```python
  def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
      ...
  
  def parse_args() -> argparse.Namespace:
      ...
  
  def detect_pose(landmarker: vision.PoseLandmarker, bgr_frame) -> Optional[PoseLandmarkerResult]:
      ...
  ```

### Docstrings
- Use triple-quoted docstrings for modules, classes, and public functions
- Module docstrings should describe the module's purpose concisely
- Keep docstrings brief and focused on "what" and "why", not "how"
- Example:
  ```python
  """
  Brush settings dataclass and colour conversion helpers.
  """
  
  def download_model(path: str = MODEL_PATH) -> None:
      """Download the MediaPipe pose landmarker model if not present."""
      ...
  ```

### Naming Conventions
- **Modules/files:** lowercase with underscores (`drawing_layer.py`, `mediapipe_utils.py`)
- **Classes:** PascalCase (`PoseDrawApp`, `BrushSettings`, `VideoProcessor`)
- **Functions/methods:** lowercase with underscores (`hex_to_bgr`, `detect_pose`)
- **Constants:** UPPERCASE with underscores (`LM_LEFT_WRIST`, `MODEL_PATH`, `VIDEO_DISPLAY_W`)
- **Private/internal:** prefix with underscore (`_rgb_to_ppm`, `_SF`, `_MONO`)

### Dataclasses
- Use `@dataclass` for simple data structures
- Include type hints for all fields
- Provide sensible defaults where appropriate
- Example:
  ```python
  @dataclass
  class BrushSettings:
      """All user-controllable drawing parameters."""
      left_color_hex:  str = DEFAULT_LEFT_COLOR
      right_color_hex: str = DEFAULT_RIGHT_COLOR
      size:            int = 8
      opacity:         float = 0.85
      smoothing:       int = 7
  ```

### Error Handling
- Use standard Python exceptions (`ValueError`, `FileNotFoundError`, etc.)
- Log errors using the `logging` module, not `print()`
- Configure logging early in the application lifecycle
- Example:
  ```python
  import logging
  
  logger = logging.getLogger(__name__)
  
  if not logging.getLogger().handlers:
      logging.basicConfig(
          level=logging.INFO,
          format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      )
  ```

### Architecture Patterns
- **Composition over inheritance:** Custom widgets are `tk.Frame` subclasses with `tk.Canvas` composition
- **Threading:** Use `threading.Thread` for background video processing; communicate via `queue.Queue`
- **Separation of concerns:** Keep UI logic (`app.py`, `ui.py`), drawing logic (`drawing_layer.py`), and ML logic (`mediapipe_utils.py`) separate

### Constants and Configuration
- Define constants in `pose_draw/constants.py`
- Use uppercase naming for true constants
- Group related constants together (model config, landmark indices, colors, dimensions)

### Comments
- Use `#` for inline comments when logic is non-obvious
- Prefer clear code over comments where possible
- Use section comments with ASCII art for major sections in long files:
  ```python
  # ── Design tokens ─────────────────────────────────────────────────────────
  ```

## File Organization

```
pose_draw/
├── app.py              # Main GUI application class (PoseDrawApp)
├── brush.py            # Brush settings and color utilities
├── constants.py        # Shared constants and configuration
├── drawing_layer.py    # Drawing canvas management
├── mediapipe_utils.py  # MediaPipe wrapper functions
├── processor.py        # Video processing thread
└── ui.py               # Custom widget components
```

## Important Notes

- **macOS Compatibility:** Widget implementation uses composition (Frame + Canvas) to avoid Tkinter crashes on macOS
- **Frame Encoding:** Uses numpy RGB → PPM bytes → tk.PhotoImage (avoiding PIL/ImageTk) for better performance
- **Thread Safety:** Video processing runs in background thread; UI updates via queue
- **Model File:** `pose_landmarker.task` (~9.4 MB) is downloaded automatically on first run

## Git Workflow

Standard ignored files are configured in `.gitignore`:
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`)
- Build artifacts (`build/`, `dist/`, `*.egg-info`)
