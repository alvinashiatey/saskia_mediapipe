"""
Shared constants: model config, landmark indices, pose skeleton, default colours.
"""

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# MediaPipe Pose landmark indices used by this app
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16

# Full skeleton connection list (from MediaPipe spec)
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),                          # face left
    (0, 4), (4, 5), (5, 6), (6, 8),                          # face right
    (9, 10),                                                   # mouth
    (11, 12),                                                  # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm
    (11, 23), (12, 24), (23, 24),                              # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),         # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),         # right leg
]

DEFAULT_LEFT_COLOR = "#00C8FF"   # cyan  — left wrist
DEFAULT_RIGHT_COLOR = "#FF6B35"   # orange — right wrist

# Video display dimensions inside the Tkinter canvas (scaled to fit)
VIDEO_DISPLAY_W = 854
VIDEO_DISPLAY_H = 480
