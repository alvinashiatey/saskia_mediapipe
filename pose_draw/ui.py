import tkinter as tk
import customtkinter as ctk
from typing import Callable


class Btn(tk.Frame):
    """CTk-backed button wrapper used by the app.

    Accepts an optional `font` parameter that should be a Tk font tuple.
    """

    def __init__(self, parent, text: str, command: Callable, *,
                 bg: str = None, bg_hi: str = None, fg: str = None,
                 font=None, width: int = 0, height: int = 36):
        super().__init__(parent, width=width or 120, height=height)
        self.pack_propagate(False)
        btn_kwargs = {
            "master": self,
            "text": text,
            "command": command,
            "fg_color": bg,
            "hover_color": bg_hi,
            "text_color": fg,
            "height": height,
            "corner_radius": 8,
        }
        if font:
            btn_kwargs["font"] = font
        self._btn = ctk.CTkButton(**btn_kwargs)
        self._btn.pack(fill="both", expand=True)

    def set_text(self, text: str):
        self._btn.configure(text=text)

    def set_bg(self, bg: str):
        self.configure(bg=bg)
        try:
            self._btn.configure(fg_color=bg)
        except Exception:
            pass


class Slider(tk.Frame):
    """Thin wrapper around `ctk.CTkSlider` exposing `get()` and `set_bg()`."""

    def __init__(self, parent, from_: float, to: float, value: float, command: Callable, *, width: int = 160):
        super().__init__(parent, width=width, height=36)
        self.pack_propagate(False)
        self._var = tk.DoubleVar(value=value)
        self._s = ctk.CTkSlider(self, from_=from_, to=to,
                                variable=self._var, command=command)
        self._s.pack(fill="x", expand=True)

    def get(self) -> float:
        return float(self._var.get())

    def set_bg(self, bg: str):
        self.configure(bg=bg)


class Toggle(tk.Frame):
    """Wrapper around `ctk.CTkSwitch`."""

    def __init__(self, parent, value: bool, command: Callable):
        super().__init__(parent, width=40, height=24)
        self.pack_propagate(False)
        self._var = tk.BooleanVar(value=value)
        self._sw = ctk.CTkSwitch(self, text="", variable=self._var,
                                 command=lambda: command(self._var.get()))
        self._sw.pack()

    def set(self, v: bool):
        self._var.set(v)

    def set_bg(self, bg: str):
        self.configure(bg=bg)


class Swatch(tk.Frame):
    """Small colour swatch implemented with `CTkButton`."""

    def __init__(self, parent, color: str, command: Callable, size: int = 28):
        super().__init__(parent, width=size, height=size)
        self.pack_propagate(False)
        self._color = color
        self._btn = ctk.CTkButton(self, text="", command=command,
                                  fg_color=color, width=size, height=size, corner_radius=6)
        self._btn.pack(fill="both", expand=True)

    def set_color(self, color: str):
        self._color = color
        try:
            self._btn.configure(fg_color=color)
        except Exception:
            pass

    def set_bg(self, bg: str):
        self.configure(bg=bg)
