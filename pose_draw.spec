# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Pose Draw macOS app bundle.
"""

from PyInstaller.utils.hooks import collect_submodules


hiddenimports = collect_submodules("mediapipe") + [
    "cv2",
]

datas = [
    ("pose_landmarker.task", "."),
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="pose_draw",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

app = BUNDLE(
    exe,
    name="Pose Draw.app",
    icon="assets/app_icon.icns",
    bundle_identifier="com.saskia.posedraw",
)
