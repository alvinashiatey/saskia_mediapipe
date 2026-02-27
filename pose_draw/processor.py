import threading
import time
import queue
import logging
from typing import Callable, Optional

import cv2
import numpy as np

from pose_draw.mediapipe_utils import (
    build_pose_landmarker,
    detect_pose,
    draw_pose_on_frame,
    get_wrist_px,
)
from pose_draw.brush import hex_to_bgr
from pose_draw.constants import LM_LEFT_WRIST, LM_RIGHT_WRIST


class VideoProcessor(threading.Thread):
    """Background thread that reads video frames, runs pose detection,
    updates `layer` (a DrawingLayer instance) and pushes frames to `frame_q`.

    The `playing_getter` is a callable returning a truthy value indicating
    whether playback should proceed. `stop_event` is a threading.Event used
    to stop the thread.
    """

    def __init__(
        self,
        video_path: str,
        frame_q: "queue.Queue",
        stop_event: threading.Event,
        playing_getter: Callable[[], bool],
        det_conf: float,
        trk_conf: float,
        brush,
        layer,
        display_size: Optional[tuple] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_q = frame_q
        self.stop_event = stop_event
        self.playing_getter = playing_getter
        self.det_conf = det_conf
        self.trk_conf = trk_conf
        self.brush = brush
        self.layer = layer
        self.display_size = display_size

    def run(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("VideoProcessor starting for %s", self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        pose = build_pose_landmarker(self.det_conf, self.trk_conf)

        # Determine pacing from source FPS (fallback to 30fps)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 0.0
        try:
            fps_src = float(fps_src) if fps_src else 30.0
        except Exception:
            fps_src = 30.0
        frame_interval = 1.0 / max(fps_src, 1e-3)

        clock = time.perf_counter()
        fidx = 0
        prev_l: Optional[tuple] = None
        prev_r: Optional[tuple] = None

        def _enqueue(item):
            try:
                self.frame_q.put_nowait(item)
            except queue.Full:
                try:
                    # drop the oldest frame to make room for the latest
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_q.put_nowait(item)
                except queue.Full:
                    logger.debug(
                        "Frame queue still full after dropping oldest")

        try:
            while not self.stop_event.is_set():
                if not self.playing_getter():
                    time.sleep(0.04)
                    continue

                ret, frame = cap.read()
                if not ret:
                    # End of file — notify UI and pause playback
                    try:
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        logger.info(
                            "VideoProcessor reached EOF (%s frames)", total)
                        _enqueue(("end", total))
                    except Exception:
                        logger.debug("Failed to enqueue end-of-file event")
                    time.sleep(0.05)
                    continue

                fidx += 1
                src_h, src_w = frame.shape[0], frame.shape[1]
                w, h = src_w, src_h
                res = detect_pose(pose, frame)

                if self.brush.show_pose and res.pose_landmarks:
                    draw_pose_on_frame(frame, res.pose_landmarks)

                if res.pose_landmarks and self.layer:
                    for lms in res.pose_landmarks:
                        lp = get_wrist_px(lms, LM_LEFT_WRIST, w, h)
                        rp = get_wrist_px(lms, LM_RIGHT_WRIST, w, h)

                        if lp is None and prev_l is not None:
                            self.layer.pen_up("left")
                        if rp is None and prev_r is not None:
                            self.layer.pen_up("right")

                        self.layer.add_point(
                            lp[0] if lp else None, lp[1] if lp else None, "left"
                        )
                        self.layer.add_point(
                            rp[0] if rp else None, rp[1] if rp else None, "right"
                        )

                        for pt, col in [(lp, self.brush.left_color_hex), (rp, self.brush.right_color_hex)]:
                            if pt:
                                cv2.circle(
                                    frame, pt, self.brush.size + 5, hex_to_bgr(col), 2)
                                cv2.circle(frame, pt, 3, (255, 255, 255), -1)
                        prev_l, prev_r = lp, rp
                else:
                    if self.layer:
                        if prev_l:
                            self.layer.pen_up("left")
                        if prev_r:
                            self.layer.pen_up("right")
                    prev_l = prev_r = None

                if self.brush.show_drawing and self.layer:
                    frame = self.layer.composite_onto(frame, self.brush)

                now = time.perf_counter()
                fps = 1.0 / max(now - clock, 1e-9)
                clock = now

                # optionally resize to the target display size to reduce GUI work,
                # preserving aspect ratio to avoid stretching
                if self.display_size:
                    try:
                        target_w, target_h = int(
                            self.display_size[0]), int(self.display_size[1])
                        if target_w > 0 and target_h > 0:
                            scale = min(target_w / src_w, target_h / src_h)
                            if scale != 1.0:
                                new_w = max(1, int(src_w * scale))
                                new_h = max(1, int(src_h * scale))
                                frame = cv2.resize(
                                    frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                w, h = new_w, new_h
                    except Exception:
                        pass

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    # send ppm bytes, frame index and fps
                    ppm = f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes()
                    _enqueue(("frame", ppm, fidx, fps))
                except Exception:
                    # be defensive: don't let queue issues crash the worker
                    logger.debug("Failed to enqueue frame %s", fidx)
                # Pace reads to approximately match source FPS when playing
                try:
                    if self.playing_getter():
                        proc_elapsed = time.perf_counter() - now
                        to_sleep = frame_interval - proc_elapsed
                        if to_sleep > 0:
                            time.sleep(to_sleep)
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Unhandled exception in VideoProcessor: %s", exc)
        finally:
            cap.release()
            logger.info("VideoProcessor stopped for %s", self.video_path)
