"""
Traductor EN->ES en tiempo real para interpretes medicos.

Captura audio del sistema (loopback WASAPI) -> Whisper -> Argos EN->ES.

Uso:
  python clinic_translate.py --setup-langs      # una vez: descarga Argos en<->es
  python clinic_translate.py                    # ejecutar
  python clinic_translate.py --list-devices     # ver dispositivos
  python clinic_translate.py --device 2         # elegir loopback especifico
  python clinic_translate.py --model small.en   # modelo mas preciso

Requisitos: Windows 10+, Python 3.10+, NVIDIA GPU recomendada.
"""
from __future__ import annotations

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# CUDA DLLs must load before faster_whisper
import app.cuda_setup  # noqa: F401

from app.audio import print_devices
from app.translation import ensure_bidirectional
from app.whisper_engine import MEDICAL_PROMPT
from app import ui


def main() -> None:
    import logging
    for n in ("huggingface_hub", "huggingface_hub.utils"):
        logging.getLogger(n).setLevel(logging.ERROR)

    p = argparse.ArgumentParser(
        description="Traductor EN->ES en tiempo real para interpretes medicos.")
    p.add_argument("--list-devices", action="store_true",
                   help="Muestra dispositivos loopback y microfonos disponibles.")
    p.add_argument("--setup-langs", action="store_true",
                   help="Descarga paquetes Argos en<->es (una vez, requiere internet).")
    p.add_argument("--model", default="base.en",
                   choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"],
                   help="Modelo Whisper. Los '.en' son mas precisos para ingles.")
    p.add_argument("--chunk-seconds", type=float, default=3.0,
                   help="Ventana de captura en segundos (default: 3).")
    p.add_argument("--max-history", type=int, default=50,
                   help="Maximo de lineas visibles por panel.")
    p.add_argument("--vad", action="store_true",
                   help="Filtro VAD de Whisper (puede omitir musica de fondo).")
    p.add_argument("--device", default=None,
                   help="Loopback: numero o nombre parcial (--list-devices para ver).")
    p.add_argument("--mic", default=None,
                   help="Microfono: numero o nombre parcial (para futuro tab ES->EN).")
    p.add_argument("--prompt", default=None,
                   help="Prompt de contexto para Whisper (default: vocabulario medico).")
    args = p.parse_args()

    if args.list_devices:
        print_devices()
        return

    if args.setup_langs:
        sys.exit(0 if ensure_bidirectional() else 1)

    if not ensure_bidirectional():
        print("Ejecuta primero: python clinic_translate.py --setup-langs", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt if args.prompt is not None else MEDICAL_PROMPT

    ui.run(
        model_size=args.model,
        chunk_sec=args.chunk_seconds,
        max_lines=args.max_history,
        vad_filter=args.vad,
        device_hint=args.device,
        mic_hint=args.mic,
        prompt=prompt,
    )


if __name__ == "__main__":
    main()
