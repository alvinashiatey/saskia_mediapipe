"""
Brush settings dataclass and colour conversion helpers.
"""

from dataclasses import dataclass

from pose_draw.constants import DEFAULT_LEFT_COLOR, DEFAULT_RIGHT_COLOR


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class BrushSettings:
    """All user-controllable drawing parameters. Mutated in-place by the UI."""

    left_color_hex:  str = DEFAULT_LEFT_COLOR
    right_color_hex: str = DEFAULT_RIGHT_COLOR
    size:            int = 8      # stroke width in pixels
    opacity:         float = 0.85   # 0.0–1.0
    smoothing:       int = 7      # rolling-average window (1 = off)
    trail_length:    int = 0      # 0 = full trail; N = last N points only
    show_pose:       bool = True   # render skeleton overlay
    show_drawing:    bool = True   # render wrist drawing layer

    @property
    def left_bgr(self) -> tuple[int, int, int]:
        return hex_to_bgr(self.left_color_hex)

    @property
    def right_bgr(self) -> tuple[int, int, int]:
        return hex_to_bgr(self.right_color_hex)
