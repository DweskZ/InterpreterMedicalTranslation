# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec para Clinic Translate (GPU - RTX 4060 / CUDA 12).

Build:
    Activar el venv, luego:
        pyinstaller clinic_translate.spec

Resultado:
    dist\ClinicTranslate\ClinicTranslate.exe
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ---------------------------------------------------------------------------
# Rutas del entorno
# ---------------------------------------------------------------------------
VENV_SITE = Path(".venv/lib/site-packages").resolve()
NVIDIA_BIN = VENV_SITE / "nvidia"

# ---------------------------------------------------------------------------
# DLLs de CUDA 12 (van junto al .exe para que cuda_setup.py las encuentre)
# ---------------------------------------------------------------------------
CUDA_DLLS = [
    ("cublas", "cublas64_12.dll"),
    ("cublas", "cublasLt64_12.dll"),
    ("cublas", "nvblas64_12.dll"),
    ("cuda_runtime", "cudart64_12.dll"),
    ("cuda_nvrtc", "nvrtc64_120_0.dll"),
    ("cuda_nvrtc", "nvrtc-builtins64_129.dll"),
]

cuda_binaries = []
for pkg, dll in CUDA_DLLS:
    src = NVIDIA_BIN / pkg / "bin" / dll
    if src.exists():
        cuda_binaries.append((str(src), "."))

# ---------------------------------------------------------------------------
# Paquetes con datas / binaries propios
# ---------------------------------------------------------------------------
datas_fw,   bins_fw,   hid_fw   = collect_all("faster_whisper")
datas_ct,   bins_ct,   hid_ct   = collect_all("ctranslate2")
datas_at,   bins_at,   hid_at   = collect_all("argostranslate")
datas_paw,  bins_paw,  hid_paw  = collect_all("pyaudiowpatch")

all_datas   = datas_fw  + datas_ct  + datas_at  + datas_paw
all_bins    = bins_fw   + bins_ct   + bins_at   + bins_paw + cuda_binaries
all_hidden  = hid_fw    + hid_ct    + hid_at    + hid_paw + [
    # tkinter (a veces no se detecta solo)
    "tkinter", "tkinter.font", "tkinter.ttk",
    # módulos propios de la app
    "app.cuda_setup", "app.audio", "app.workers",
    "app.whisper_engine", "app.translation", "app.ui",
    # scipy (se usa para resample)
    "scipy.signal", "scipy._lib.messagestream",
    "scipy.special._ufuncs", "scipy.special._cython_special",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["clinic_translate.py"],
    pathex=["."],
    binaries=all_bins,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "PIL", "cv2", "IPython", "jupyter",
        "notebook", "pandas", "sklearn", "tensorflow",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClinicTranslate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX puede romper DLLs CUDA — desactivado
    console=True,       # Cambiar a False cuando todo funcione bien
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ClinicTranslate",
)
