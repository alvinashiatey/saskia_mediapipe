"""
DrawingLayer — accumulates wrist paths, renders them as a BGRA overlay,
alpha-composites onto BGR frames, and exports to PNG / SVG / JSON.
"""

import json
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from pose_draw.brush import BrushSettings


@dataclass
class DrawingLayer:
    """
    Stores left- and right-wrist paths as lists of (x, y) | None.
    None entries act as pen-lifts so disconnected strokes stay separate.

    All public methods are thread-safe via an internal RLock.
    """

    width:  int
    height: int
    canvas: np.ndarray = field(init=False)
    left_path:  list = field(default_factory=list)
    right_path: list = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    # ── path recording ────────────────────────────────────────────────────────

    def add_point(self, x: Optional[int], y: Optional[int], wrist: str) -> None:
        with self._lock:
            path = self.left_path if wrist == "left" else self.right_path
            path.append((x, y) if x is not None else None)

    def pen_up(self, wrist: str) -> None:
        """Insert a None break to prevent connecting across occluded frames."""
        with self._lock:
            path = self.left_path if wrist == "left" else self.right_path
            if path and path[-1] is not None:
                path.append(None)

    def clear(self) -> None:
        with self._lock:
            self.left_path.clear()
            self.right_path.clear()
            self.canvas[:] = 0

    # ── smoothing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(pts: list, window: int) -> list:
        """Apply a centred rolling average to valid (non-None) points."""
        if window < 2 or len(pts) < 2:
            return pts
        valid = [(i, p) for i, p in enumerate(pts) if p is not None]
        smooth_map: dict[int, tuple[int, int]] = {}
        for j, (i, _) in enumerate(valid):
            lo = max(0, j - window // 2)
            hi = min(len(valid), j + window // 2 + 1)
            xs = [valid[k][1][0] for k in range(lo, hi)]
            ys = [valid[k][1][1] for k in range(lo, hi)]
            smooth_map[i] = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
        return [smooth_map[i] if p is not None else None for i, p in enumerate(pts)]

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self, brush: BrushSettings) -> np.ndarray:
        """Redraw the internal BGRA canvas from the current paths and brush."""
        with self._lock:
            self.canvas[:] = 0
            pairs = [
                (list(self.left_path),  brush.left_bgr),
                (list(self.right_path), brush.right_bgr),
            ]

        for raw_path, color_bgr in pairs:
            if brush.trail_length > 0:
                valid_pts = [p for p in raw_path if p is not None]
                raw_path = valid_pts[-brush.trail_length:]

            path = self._smooth(raw_path, brush.smoothing)
            alpha_val = int(brush.opacity * 255)
            prev: Optional[tuple[int, int]] = None

            for pt in path:
                if pt is None:
                    prev = None
                    continue
                if prev is not None:
                    cv2.line(
                        self.canvas, prev, pt,
                        (*color_bgr, alpha_val),
                        brush.size, cv2.LINE_AA,
                    )
                cv2.circle(
                    self.canvas, pt, max(1, brush.size // 2),
                    (*color_bgr, alpha_val), -1, cv2.LINE_AA,
                )
                prev = pt

        return self.canvas

    # ── compositing ───────────────────────────────────────────────────────────

    def composite_onto(self, bgr_frame: np.ndarray, brush: BrushSettings) -> np.ndarray:
        """Return a new BGR frame with the drawing layer alpha-composited on top."""
        overlay = self.render(brush)
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        color = overlay[:, :, :3].astype(np.float32)
        base = bgr_frame.astype(np.float32)
        result = base * (1.0 - alpha) + color * alpha
        return result.clip(0, 255).astype(np.uint8)

    # ── export ────────────────────────────────────────────────────────────────

    def export_png(
        self,
        path: str,
        brush: BrushSettings,
        background: str = "black",
    ) -> None:
        """Export the drawing as a PNG.  background can be 'black', 'white', or 'transparent'."""
        canvas = self.render(brush)
        if background == "transparent":
            rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
            PILImage.fromarray(rgba).save(path)
        else:
            bg_rgb = (255, 255, 255) if background == "white" else (0, 0, 0)
            bg = np.full((self.height, self.width, 3), bg_rgb, dtype=np.uint8)
            alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
            color = canvas[:, :, :3].astype(np.float32)
            flat = (bg.astype(np.float32) * (1 - alpha) +
                    color * alpha).clip(0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
            PILImage.fromarray(rgb).save(path)
        print(f"[export] PNG → {path}")

    def export_svg(self, path: str, brush: BrushSettings) -> None:
        """Export the drawing as an SVG with one <polyline> per stroke segment."""
        with self._lock:
            left_copy = list(self.left_path)
            right_copy = list(self.right_path)

        def segments(pts: list, window: int) -> list[list[tuple[int, int]]]:
            smoothed = self._smooth(pts, window)
            result, current = [], []
            for pt in smoothed:
                if pt is None:
                    if current:
                        result.append(current)
                    current = []
                else:
                    current.append(pt)
            if current:
                result.append(current)
            return result

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">',
            f'  <rect width="{self.width}" height="{self.height}" fill="black"/>',
        ]

        for pts, hex_color, label in [
            (left_copy,  brush.left_color_hex,  "left-wrist"),
            (right_copy, brush.right_color_hex, "right-wrist"),
        ]:
            for seg in segments(pts, brush.smoothing):
                if len(seg) < 2:
                    continue
                pts_str = " ".join(f"{x},{y}" for x, y in seg)
                lines.append(
                    f'  <polyline class="{label}" points="{pts_str}" '
                    f'stroke="{hex_color}" stroke-width="{brush.size}" '
                    f'stroke-linecap="round" stroke-linejoin="round" '
                    f'fill="none" opacity="{brush.opacity:.2f}"/>'
                )

        lines.append("</svg>")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[export] SVG → {path}")

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        with self._lock:
            return {
                "width":      self.width,
                "height":     self.height,
                "left_path":  [[p[0], p[1]] if p else None for p in self.left_path],
                "right_path": [[p[0], p[1]] if p else None for p in self.right_path],
            }

    def from_json(self, data: dict) -> None:
        with self._lock:
            self.left_path = [
                tuple(p) if p else None for p in data.get("left_path",  [])]
            self.right_path = [
                tuple(p) if p else None for p in data.get("right_path", [])]
            self.canvas = np.zeros(
                (self.height, self.width, 4), dtype=np.uint8)
