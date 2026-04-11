"""
pre_download.py — Descarga modelos Whisper y paquetes Argos EN<->ES.

Corre ANTES del build de PyInstaller para que los modelos queden en
las carpetas correctas junto al ejecutable.

Uso:
    .venv\Scripts\python.exe scripts\pre_download.py [modelo]

    modelo: tiny.en | base.en | small.en | medium.en  (default: base.en)

Las carpetas se crean en:
    dist_assets\models\         <- modelos Whisper (HF cache)
    dist_assets\argos-packages\ <- paquetes de traduccion Argos
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas de destino (junto al repo, para copiarlas en el build)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "dist_assets"
MODELS_DIR = ASSETS / "models"
ARGOS_DIR = ASSETS / "argos-packages"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARGOS_DIR.mkdir(parents=True, exist_ok=True)

# Apuntar las librerías a nuestras carpetas ANTES de importarlas
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["ARGOS_PACKAGES_DIR"] = str(ARGOS_DIR)


def download_whisper(model_size: str) -> None:
    print(f"\n[1/2] Descargando modelo Whisper '{model_size}'...")
    print("      (puede tardar varios minutos la primera vez)")
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device="cpu", compute_type="int8")
    # smoke test para confirmar que está bien
    import numpy as np
    segs, _ = m.transcribe(
        np.zeros(16000, dtype=np.float32),
        language="en", beam_size=1, vad_filter=False, without_timestamps=True,
    )
    list(segs)
    print(f"      ✓ Modelo '{model_size}' listo en: {MODELS_DIR}")


def download_argos() -> None:
    print("\n[2/2] Descargando paquetes de traduccion Argos EN<->ES...")
    import argostranslate.package as pkg

    print("      Actualizando indice...")
    pkg.update_package_index()
    available = pkg.get_available_packages()

    for src, tgt in [("en", "es"), ("es", "en")]:
        match = next(
            (p for p in available if p.from_code == src and p.to_code == tgt), None
        )
        if match is None:
            print(f"      ✗ No se encontro paquete {src}->{tgt} en el indice")
            continue
        already = pkg.get_installed_packages()
        if any(p.from_code == src and p.to_code == tgt for p in already):
            print(f"      ✓ {src}->{tgt} ya instalado")
            continue
        print(f"      Descargando {src}->{tgt}...")
        pkg.install_from_path(match.download())
        print(f"      ✓ {src}->{tgt} instalado en: {ARGOS_DIR}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "base.en"

    valid = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"]
    if model not in valid:
        print(f"Modelo invalido '{model}'. Opciones: {', '.join(valid)}")
        sys.exit(1)

    print("=" * 60)
    print(" Clinic Translate — Pre-descarga de assets")
    print("=" * 60)

    download_whisper(model)
    download_argos()

    print("\n" + "=" * 60)
    print(" ✓ Todo listo. Ahora puedes correr scripts\\build.bat")
    print("=" * 60)
