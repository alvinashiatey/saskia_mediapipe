"""
PoseDrawApp — macOS-safe Tkinter UI.

Widget strategy: composition over inheritance.
  Each custom widget (button, slider, toggle, swatch) is a tk.Frame subclass
  that owns a tk.Canvas internally.  The Frame is registered with Tk first,
  then the Canvas is created inside it — both are fully registered before
  any drawing call is made.  This eliminates the "invalid command name N"
  crash that occurs on macOS when tk.Canvas subclasses call self.delete()
  before their Tcl command is registered.

Frame rendering: numpy RGB → PPM bytes → tk.PhotoImage (no PIL/ImageTk).
"""

import os
import queue
import threading
import time
import signal
import logging
import tkinter as tk
import tkinter.font as tkfont
import customtkinter as ctk
from pose_draw.ui import Btn, Slider, Toggle, Swatch
from tkinter import colorchooser, filedialog, messagebox
from typing import Callable, Optional

import cv2
import numpy as np

from pose_draw.brush import BrushSettings, hex_to_bgr
from pose_draw.constants import (
    LM_LEFT_WRIST,
    LM_RIGHT_WRIST,
    VIDEO_DISPLAY_W,
    VIDEO_DISPLAY_H,
)
from pose_draw.drawing_layer import DrawingLayer
from pose_draw.mediapipe_utils import (
    download_model,
)
from pose_draw.processor import VideoProcessor

logger = logging.getLogger(__name__)

# Configure a sane default logging if the application hasn't configured logging yet
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Design tokens ─────────────────────────────────────────────────────────────

BG = "#0F0F0F"
SIDEBAR = "#161616"
CARD = "#1E1E1E"
CARD_HI = "#272727"
RULE = "#2A2A2A"
FG1 = "#F2F2F2"   # primary text
FG2 = "#B8B8B8"   # secondary text
FG3 = "#8E8E8E"   # muted/meta text (still readable)
ACCENT = "#3B9EF5"
ACCENT_HI = "#5BB3FF"
DANGER = "#E05252"
DANGER_HI = "#FF6B6B"

# Typography tokens
_SF = "SF Pro Text"
_MONO = "Menlo"

FS_TITLE = 18
FS_SUBTITLE = 11
FS_SECTION = 10
FS_BODY = 12
FS_META = 10
FS_VALUE = 10


def _f(size: int, weight: str = "normal", family: str = _SF):
    return (family, size, weight)


# ─────────────────────────────────────────────────────────────────────────────
# Custom widgets — Frame-based composition
# Each class is a tk.Frame.  The Canvas lives *inside* the Frame and is
# created only after super().__init__ has fully registered the Frame with Tk.
# ─────────────────────────────────────────────────────────────────────────────

# UI widgets have been moved to pose_draw.ui for better structure


# ── PPM frame encoding ────────────────────────────────────────────────────────

def _rgb_to_ppm(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    return f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes()


# ── Main application ──────────────────────────────────────────────────────────

class PoseDrawApp:

    _POLL_MS = 30

    def __init__(
        self,
        initial_video: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._video_path: Optional[str] = initial_video

        self.brush = BrushSettings()
        self.layer: Optional[DrawingLayer] = None

        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._worker:  Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._playing = False
        self._frame_idx = 0
        self._total_frames = 0
        self._photo_ref = None

        self._root_setup()
        self._build_ui()

        if initial_video:
            self.root.after(200, lambda: self._load_video(initial_video))

    # ── Root ──────────────────────────────────────────────────────────────────

    def _root_setup(self):
        # Use customtkinter root for modern themed widgets
        ctk.set_appearance_mode("dark")
        try:
            ctk.set_default_color_theme("dark-blue")
        except Exception:
            pass
        self.root = ctk.CTk()
        logger.info("Starting TraceDraw UI")
        self.root.title("Trace Draw")
        self.root.geometry("1300x820")
        self.root.minsize(960, 600)
        # keep original background usage for canvas-based areas
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Install signal handlers for graceful shutdown (if supported)
        try:
            def _sigterm_handler(signum, frame):
                logger.info("Received signal %s, shutting down", signum)
                try:
                    self._on_close()
                except Exception:
                    pass

            signal.signal(signal.SIGINT, _sigterm_handler)
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except Exception:
            logger.debug("Signal handlers not installed on this platform")

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Toast bar at very bottom (packed first so it's reserved before expand)
        self._toast_bar = tk.Frame(self.root, bg="#0D0D0D", height=34)
        self._toast_bar.pack(side="bottom", fill="x")
        self._toast_bar.pack_propagate(False)
        tk.Frame(self.root, bg=RULE, height=1).pack(side="bottom", fill="x")

        self._toast_c = tk.Canvas(self._toast_bar, bg="#0D0D0D",
                                  highlightthickness=0)
        self._toast_c.pack(fill="both", expand=True, padx=16)
        self._toast_id = self._toast_c.create_text(
            0, 17, anchor="w", text="", fill=FG2, font=_f(10))

        # Main row (sidebar + rule + viewport)
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        self._sidebar = tk.Frame(main, bg=SIDEBAR, width=300)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        tk.Frame(main, bg=RULE, width=1).pack(side="left", fill="y")

        self._viewport = tk.Frame(main, bg=BG)
        self._viewport.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_viewport()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = self._sidebar
        P = 16   # horizontal padding

        # Logo
        logo = tk.Frame(sb, bg="#0D0D0D", height=68)
        logo.pack(fill="x")
        logo.pack_propagate(False)
        lc = tk.Canvas(logo, bg="#0D0D0D", highlightthickness=0)
        lc.pack(fill="both", expand=True, padx=P)
        lc.create_text(0, 28, anchor="w", text="Trace Draw",
                       fill=FG1, font=_f(16, "bold"))
        lc.create_text(0, 50, anchor="w", text="Tracing the invisible labor within graphic design.",
                       fill=FG3, font=_f(10))
        tk.Frame(sb, bg=RULE, height=1).pack(fill="x")

        # Scrollable content (fill="both", expand=True — fills all remaining space)
        sc = tk.Canvas(sb, bg=SIDEBAR, highlightthickness=0)
        sc.pack(fill="both", expand=True)

        inner = tk.Frame(sc, bg=SIDEBAR)
        win = sc.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.bind("<Configure>",
                lambda e: sc.itemconfig(win, width=e.width))
        # Trackpad / mousewheel scroll

        def _on_wheel(e):
            d = e.delta
            if d == 0:
                return
            # step = -1 if d > 0 else 1
            units = max(1, abs(d) // 60)
            # sc.yview_scroll(step, "units")
            sc.yview_scroll(-units if d > 0 else units, "units")

        def _bind_wheel(_=None):
            sc.bind_all("<MouseWheel>", _on_wheel)  # macOS/Windows
            # Linux up
            sc.bind_all("<Button-4>", lambda e: sc.yview_scroll(-1, "units"))
            # Linux down
            sc.bind_all("<Button-5>", lambda e: sc.yview_scroll(1, "units"))

        def _unbind_wheel(_=None):
            sc.unbind_all("<MouseWheel>")
            sc.unbind_all("<Button-4>")
            sc.unbind_all("<Button-5>")

        sc.bind("<Enter>", _bind_wheel)
        sc.bind("<Leave>", _unbind_wheel)
        inner.bind("<Enter>", _bind_wheel)
        inner.bind("<Leave>", _unbind_wheel)
        f = inner

        # ── FILE ──────────────────────────────────────────────────────────────
        self._heading(f, "FILE")

        # (Removed sidebar filename/status — viewport shows filename already)

        btn_row = tk.Frame(f, bg=SIDEBAR)

        btn_row.pack(fill="x", padx=16, pady=3)
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(0, weight=1)

        open_btn = Btn(
            btn_row,
            "Open Video…",
            self._open_video_dialog,
            bg=ACCENT,
            bg_hi=ACCENT_HI,
            fg=FG1,
            font=_f(12, "bold"),
        )
        open_btn.set_bg(CARD)
        open_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        # ── PLAYBACK ──────────────────────────────────────────────────────────
        self._heading(f, "PLAYBACK")

        pb = tk.Frame(f, bg=SIDEBAR)
        pb.pack(fill="x", padx=P, pady=(0, 8))

        BTN_W = 44
        BTN_H = 34
        GAP = 8

        self._start_btn = Btn(
            pb, "⏮", self._go_to_start,
            bg=CARD, bg_hi=CARD_HI, fg=FG1, font=_f(13, "bold"),
            width=BTN_W, height=BTN_H
        )
        self._start_btn.set_bg(SIDEBAR)
        self._start_btn.pack(side="left")

        self._play_btn = Btn(
            pb, "▶", self._toggle_play,
            bg=CARD, bg_hi=CARD_HI, fg=FG1, font=_f(13, "bold"),
            width=BTN_W, height=BTN_H
        )
        self._play_btn.set_bg(SIDEBAR)
        self._play_btn.pack(side="left", padx=(GAP, 0))

        self._stop_btn = Btn(
            pb, "⏹", self._stop_to_start,
            bg=CARD, bg_hi=CARD_HI, fg=FG1, font=_f(13, "bold"),
            width=BTN_W, height=BTN_H
        )
        self._stop_btn.set_bg(SIDEBAR)
        self._stop_btn.pack(side="left", padx=(GAP, 0))

        # Progress bar
        prog = tk.Canvas(f, bg=SIDEBAR, highlightthickness=0, height=8)
        prog.pack(fill="x", padx=P, pady=(0, 4))
        self._prog_c = prog
        self._prog_bg = prog.create_rectangle(
            0, 2, 2000, 6, fill=CARD, outline="")
        self._prog_fill = prog.create_rectangle(
            0, 2, 0,    6, fill=ACCENT, outline="")

        fc = tk.Canvas(f, bg=SIDEBAR, highlightthickness=0, height=16)
        fc.pack(fill="x", padx=P, pady=(0, 8))
        self._frame_c = fc
        self._frame_id = fc.create_text(
            2000, 8, anchor="e", text="— / —", fill=FG3, font=_f(10, family=_MONO))
        fc.bind("<Configure>",
                lambda e: fc.coords(self._frame_id, e.width - 2, 8))

        # ── BRUSH ─────────────────────────────────────────────────────────────
        self._heading(f, "BRUSH")

        brush_card = tk.Frame(f, bg=CARD)
        brush_card.pack(fill="x", padx=P, pady=(0, 2))

        self._sliders: dict[str, Slider] = {}
        for label, attr, lo, hi, val in [
            ("Size",         "size",         1.0,  40.0, float(self.brush.size)),
            ("Opacity",      "opacity",      0.05,  1.0, self.brush.opacity),
            ("Smoothing",    "smoothing",    1.0,
             30.0, float(self.brush.smoothing)),
            ("Trail  (0=∞)", "trail_length", 0.0,
             300.0, float(self.brush.trail_length)),
        ]:
            self._slider_row(brush_card, label, attr, lo, hi, val)

        # ── COLOURS ───────────────────────────────────────────────────────────
        self._heading(f, "COLOURS")

        card = tk.Frame(f, bg=CARD)
        card.pack(fill="x", padx=P, pady=(0, 6))

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=14, pady=10)

        # Let label columns expand; swatch columns stay compact
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=0)
        row.columnconfigure(2, weight=1)
        row.columnconfigure(3, weight=0)

        left_lbl = tk.Label(row, text="LW",
                            bg=CARD, fg=FG1, font=_f(12))
        left_lbl.grid(row=0, column=0, sticky="w")

        def left_cb(): return self._pick_color("left")

        left_sw = Swatch(row, self.brush.left_color_hex, left_cb, size=28)
        left_sw.set_bg(CARD)
        left_sw.grid(row=0, column=1, sticky="e", padx=(8, 16))
        self._left_swatch = left_sw

        right_lbl = tk.Label(row, text="RW",
                             bg=CARD, fg=FG1, font=_f(12))
        right_lbl.grid(row=0, column=2, sticky="w")

        def right_cb(): return self._pick_color("right")

        right_sw = Swatch(row, self.brush.right_color_hex, right_cb, size=28)
        right_sw.set_bg(CARD)
        right_sw.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self._right_swatch = right_sw

        # ── VISIBILITY ────────────────────────────────────────────────────────
        self._heading(f, "VISIBILITY")

        card = tk.Frame(f, bg=CARD)
        card.pack(fill="x", padx=P, pady=(0, 6))

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=14, pady=10)

        # Label columns expand; toggle columns stay compact
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=0)
        row.columnconfigure(2, weight=1)
        row.columnconfigure(3, weight=0)

        pose_lbl = tk.Label(row, text="Pose",
                            bg=CARD, fg=FG1, font=_f(12))
        pose_lbl.grid(row=0, column=0, sticky="w")

        pose_tgl = Toggle(
            row,
            value=getattr(self.brush, "show_pose", True),
            command=lambda v: setattr(self.brush, "show_pose", v),
        )
        pose_tgl.set_bg(CARD)
        pose_tgl.grid(row=0, column=1, sticky="e", padx=(8, 16))
        self._pose_toggle = pose_tgl

        draw_lbl = tk.Label(row, text="Draw",
                            bg=CARD, fg=FG1, font=_f(12))
        draw_lbl.grid(row=0, column=2, sticky="w")

        draw_tgl = Toggle(
            row,
            value=getattr(self.brush, "show_drawing", True),
            command=lambda v: setattr(self.brush, "show_drawing", v),
        )
        draw_tgl.set_bg(CARD)
        draw_tgl.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self._drawing_toggle = draw_tgl

        # ── EXPORT (compact) ──────────────────────────────────────────────────
        self._heading(f, "EXPORT")

        erow = tk.Frame(f, bg=CARD)
        erow.pack(fill="x", padx=16, pady=(4, 6))

        png_btn = Btn(erow, "PNG", self._export_png,
                      bg=CARD, bg_hi=CARD_HI, fg=FG1,
                      font=_f(FS_BODY), width=48, height=36)
        png_btn.set_bg(CARD)
        png_btn.pack(side="left")

        svg_btn = Btn(erow, "SVG", self._export_svg,
                      bg=CARD, bg_hi=CARD_HI, fg=FG1,
                      font=_f(FS_BODY), width=48, height=36)
        svg_btn.set_bg(CARD)
        svg_btn.pack(side="left", padx=(8, 0))

        # small textual hints to the right
        hint = tk.Label(erow, text="Export format: PNG · SVG",
                        bg=CARD, fg=FG2, font=_f(FS_META))
        hint.pack(side="left", padx=(12, 0))

        self._row_btn(f, "Clear Drawing", self._clear_drawing,
                      bg=DANGER, bg_hi=DANGER_HI, fg=FG1)

        tk.Frame(f, bg=SIDEBAR, height=24).pack()   # bottom padding

    # ── Viewport ──────────────────────────────────────────────────────────────

    def _build_viewport(self):
        vp = self._viewport

        bar = tk.Frame(vp, bg="#0C0C0C", height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        self._name_c = tk.Canvas(bar, bg="#0C0C0C", highlightthickness=0)
        self._name_c.pack(side="left", fill="both", expand=True, padx=16)
        self._name_id = self._name_c.create_text(
            0, 22, anchor="w", text="No video", fill=FG2, font=_f(11))

        self._fps_c = tk.Canvas(bar, bg="#0C0C0C", highlightthickness=0,
                                width=80)
        self._fps_c.pack(side="right", padx=16)
        self._fps_id = self._fps_c.create_text(
            80, 22, anchor="e", text="", fill=FG3, font=_f(10, family=_MONO))

        tk.Frame(vp, bg=RULE, height=1).pack(fill="x")

        self._video_canvas = tk.Canvas(vp, bg="#000000", highlightthickness=0)
        self._video_canvas.pack(fill="both", expand=True)
        self._video_canvas.bind(
            "<Configure>",
            lambda _: self._draw_placeholder() if not self._photo_ref else None)
        self._draw_placeholder()

    def _draw_placeholder(self):
        c = self._video_canvas
        cw = c.winfo_width() or VIDEO_DISPLAY_W
        ch = c.winfo_height() or VIDEO_DISPLAY_H
        c.delete("all")
        step = 44
        for x in range(step, cw, step):
            for y in range(step, ch, step):
                c.create_oval(x-1, y-1, x+1, y+1, fill="#1C1C1C", outline="")
        c.create_text(cw//2, ch//2 - 14, text="No video loaded",
                      fill=FG3, font=_f(15, "bold"))
        c.create_text(cw//2, ch//2 + 14,
                      text="Click  Open Video…  in the sidebar",
                      fill=FG3, font=_f(11))

    # ── Widget factory helpers ────────────────────────────────────────────────

    def _heading(self, parent: tk.Frame, title: str):
        tk.Frame(parent, bg=SIDEBAR, height=14).pack(fill="x")
        c = tk.Canvas(parent, bg=SIDEBAR, highlightthickness=0, height=14)
        c.pack(fill="x", padx=16)
        c.create_text(0, 7, anchor="w", text=title,
                      fill=FG3, font=_f(10, "bold", _MONO))
        tk.Frame(parent, bg=RULE, height=1).pack(
            fill="x", padx=16, pady=(2, 6))

    def _row_btn(self, parent, text, cmd, *,
                 bg=CARD, bg_hi=CARD_HI, fg=FG1) -> Btn:
        """Full-width button row in the sidebar.

        Create the button with an idle background matching the sidebar and
        the provided hover color so the button appears flat until hovered.
        """
        # Construct with the action color then set the idle bg to the sidebar
        # so the button appears flat until hovered (matches Open Video button).
        btn = Btn(parent, text, cmd, bg=bg, bg_hi=bg_hi, fg=fg,
                  font=_f(12, "bold"))
        btn.set_bg(SIDEBAR)
        btn.pack(fill="x", padx=16, pady=3)
        return btn

    def _slider_row(self, parent, label, attr, lo, hi, val):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=12, pady=1)   # tightened vertical spacing

        lbl = tk.Canvas(row, bg=CARD, highlightthickness=0,
                        width=92, height=20)
        lbl.pack(side="left")
        lbl.create_text(0, 10, anchor="w", text=label,
                        fill=FG2, font=_f(FS_META))

        fmt = "{:.2f}" if (hi - lo) <= 2 else "{:.0f}"
        val_c = tk.Canvas(row, bg=CARD, highlightthickness=0,
                          width=40, height=20)
        val_c.pack(side="right")
        val_id = val_c.create_text(
            40, 10, anchor="e", text=fmt.format(val), fill=FG2, font=_f(FS_VALUE, family=_MONO)
        )

        def _cb(v, _c=val_c, _id=val_id, _fmt=fmt, _a=attr):
            _c.itemconfig(_id, text=_fmt.format(float(v)))
            fv = float(v)
            setattr(self.brush, _a,
                    int(round(fv)) if _a in ("size", "smoothing", "trail_length") else fv)

        sl = Slider(row, from_=lo, to=hi, value=val, command=_cb, width=110)
        sl.set_bg(CARD)
        sl.pack(side="left", padx=(6, 0))

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _open_video_dialog(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"),
                       ("All files", "*.*")])
        if path:
            self._load_video(path)

    # ── Video loading ─────────────────────────────────────────────────────────

    def _load_video(self, path: str):
        self._stop_worker()
        self._playing = False
        self._video_path = path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return

        self._video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Prepare drawing layer before showing first frame
        self.layer = DrawingLayer(width=self._video_w, height=self._video_h)

        # Show the first frame immediately (if available)
        first_shown = False
        ret, first_frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            try:
                self._show_frame(_rgb_to_ppm(rgb))
                self._frame_idx = 1
                self._frame_c.itemconfig(
                    self._frame_id, text=f"1 / {self._total_frames}")
                w = self._prog_c.winfo_width()
                pct = 1 / max(self._total_frames, 1)
                self._prog_c.coords(self._prog_fill, 0, 2, int(w * pct), 6)
                first_shown = True
            except Exception:
                first_shown = False
        cap.release()

        name = os.path.basename(path)
        self._name_c.itemconfig(self._name_id, text=name)
        # If we did not successfully show the first frame, draw the placeholder
        if not first_shown:
            self._frame_c.itemconfig(
                self._frame_id, text=f"0 / {self._total_frames}")
            self._draw_placeholder()
        else:
            # keep the first-frame counter shown
            self._frame_c.itemconfig(
                self._frame_id, text=f"{self._frame_idx} / {self._total_frames}")
        self._play_btn.set_text("▶")

        download_model()
        self._start_worker()
        self._toast("▶")

    def _start_worker(self):
        self._stop.clear()
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break
        logger.info("Starting video processor for %s", self._video_path)
        # Create and start the background video processor thread
        # capture current display canvas size as a hint for the worker to
        # resize frames before enqueueing (reduces GUI work).
        try:
            cw = max(1, int(self._video_canvas.winfo_width()))
            ch = max(1, int(self._video_canvas.winfo_height()))
        except Exception:
            cw, ch = VIDEO_DISPLAY_W, VIDEO_DISPLAY_H

        self._worker = VideoProcessor(
            self._video_path,
            self._frame_q,
            self._stop,
            lambda: self._playing,
            self._det_conf,
            self._trk_conf,
            self.brush,
            self.layer,
            display_size=(cw, ch),
        )
        self._worker.start()
        self.root.after(self._POLL_MS, self._poll)

    def _stop_worker(self):
        logger.info("Stopping video processor")
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None

    # ── Processing loop ───────────────────────────────────────────────────────

    # Processing loop extracted to pose_draw.processor.VideoProcessor

    # ── Frame polling ─────────────────────────────────────────────────────────

    def _poll(self):
        try:
            item = self._frame_q.get_nowait()
        except queue.Empty:
            item = None

        if item:
            if item[0] == "frame":
                _, ppm, fidx, fps = item
                self._show_frame(ppm)
                self._frame_idx = fidx
                pct = fidx / max(self._total_frames, 1)
                w = self._prog_c.winfo_width()
                self._prog_c.coords(self._prog_fill, 0, 2, int(w * pct), 6)
                self._frame_c.itemconfig(
                    self._frame_id, text=f"{fidx} / {self._total_frames}")
                self._fps_c.itemconfig(self._fps_id, text=f"{fps:.0f} fps")
            elif item[0] == "end":
                _, end_idx = item
                # mark playback stopped and show final progress
                self._playing = False
                w = self._prog_c.winfo_width()
                self._prog_c.coords(self._prog_fill, 0, 2, w, 6)
                self._frame_idx = end_idx
                self._frame_c.itemconfig(
                    self._frame_id, text=f"{end_idx} / {self._total_frames}")
                try:
                    self._play_btn.set_text("▶")
                except Exception:
                    pass

        if not self._stop.is_set():
            self.root.after(self._POLL_MS, self._poll)

    def _show_frame(self, ppm: bytes):
        i0 = ppm.index(b"\n")
        i1 = ppm.index(b"\n", i0 + 1)
        i2 = ppm.index(b"\n", i1 + 1)
        src_w, src_h = (int(x) for x in ppm[i0+1:i1].split())
        raw = ppm[i2+1:]

        cw = self._video_canvas.winfo_width()
        ch = self._video_canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        scale = min(cw / src_w, ch / src_h)
        dw, dh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3)
        resized = cv2.resize(arr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        photo = tk.PhotoImage(data=_rgb_to_ppm(resized))

        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self._video_canvas.delete("frame")
        self._video_canvas.create_image(ox, oy, anchor="nw",
                                        image=photo, tags="frame")
        self._photo_ref = photo

    # ── Playback ──────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if not self._video_path:
            messagebox.showinfo("No video", "Open a video file first.")
            return
        # If we're at the end, reset playback to start before playing again
        if not self._playing and self._frame_idx >= max(self._total_frames, 1):
            self._reset_playback()
        self._playing = not self._playing
        self._play_btn.set_text("⏸" if self._playing else "▶")

    def _go_to_start(self):
        self._reset_playback()

    def _stop_to_start(self):
        self._playing = False
        self._reset_playback()

    def _reset_playback(self):
        was = self._playing
        self._playing = False
        self._stop_worker()
        if self.layer:
            self.layer.pen_up("left")
            self.layer.pen_up("right")
        self._frame_idx = 0
        self._prog_c.coords(self._prog_fill, 0, 2, 0, 6)
        self._frame_c.itemconfig(
            self._frame_id, text=f"0 / {self._total_frames}")
        self._play_btn.set_text("▶")
        if self._video_path:
            self._start_worker()
            if was:
                self._playing = True
                self._play_btn.set_text("⏸")

    # ── Colour picker ─────────────────────────────────────────────────────────

    def _pick_color(self, wrist: str):
        cur = (self.brush.left_color_hex if wrist == "left"
               else self.brush.right_color_hex)
        res = colorchooser.askcolor(
            color=cur, title=f"Pick {wrist} wrist colour")
        if res and res[1]:
            hx = res[1]
            if wrist == "left":
                self.brush.left_color_hex = hx
                self._left_swatch.set_color(hx)
            else:
                self.brush.right_color_hex = hx
                self._right_swatch.set_color(hx)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _clear_drawing(self):
        if messagebox.askyesno("Clear", "Erase the drawing layer?"):
            if self.layer:
                self.layer.clear()
            self._toast("Drawing cleared")

    # ── Export ────────────────────────────────────────────────────────────────

    def _require_layer(self) -> bool:
        if not self.layer:
            messagebox.showwarning("No drawing", "Load a video first.")
            return False
        return True

    def _export_png(self):
        if not self._require_layer():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="drawing.png",
            filetypes=[("PNG image", "*.png")])
        if path:
            self.layer.export_png(path, self.brush)
            self._toast(f"PNG saved → {os.path.basename(path)}")

    def _export_svg(self):
        if not self._require_layer():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".svg", initialfile="drawing.svg",
            filetypes=[("SVG vector", "*.svg")])
        if path:
            self.layer.export_svg(path, self.brush)
            self._toast(f"SVG saved → {os.path.basename(path)}")

    # ── Toast ─────────────────────────────────────────────────────────────────

    def _toast(self, msg: str, ms: int = 3500):
        self._toast_c.itemconfig(self._toast_id, text=msg)
        self.root.after(ms, lambda: self._toast_c.itemconfig(
            self._toast_id, text=""))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        logger.info("Application closing")
        self._playing = False
        self._stop_worker()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down")
            self._on_close()
