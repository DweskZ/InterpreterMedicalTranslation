"""
Traductor EN<->ES en tiempo real para intérpretes médicos.

Captura audio del sistema (loopback WASAPI) -> Transcripción -> Traducción natural.

Uso:
  python clinic_translate.py --setup-langs      # una vez: descarga Argos en<->es
  python clinic_translate.py                    # ejecutar (Whisper local)
  python clinic_translate.py --backend deepgram # usar Deepgram (requiere DEEPGRAM_API_KEY en .env)
  python clinic_translate.py --list-devices     # ver dispositivos
  python clinic_translate.py --device 2         # elegir loopback específico
  python clinic_translate.py --model small.en   # modelo más preciso

Requisitos: Windows 10+, Python 3.10+, NVIDIA GPU recomendada.
"""
from __future__ import annotations

import argparse
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Cargar variables de entorno desde .env (si existe) antes de cualquier import
def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

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
        description="Traductor EN<->ES en tiempo real para intérpretes médicos.")
    p.add_argument("--list-devices", action="store_true",
                   help="Muestra dispositivos loopback y micrófonos disponibles.")
    p.add_argument("--setup-langs", action="store_true",
                   help="Descarga paquetes Argos en<->es (una vez, requiere internet).")
    p.add_argument("--backend", default="whisper", choices=["whisper", "deepgram"],
                   help=(
                       "Motor de transcripción: 'whisper' (local, default) o "
                       "'deepgram' (cloud, detecta EN y ES automáticamente). "
                       "Deepgram requiere DEEPGRAM_API_KEY en el archivo .env."
                   ))
    p.add_argument("--model", default="base.en",
                   choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                            "medium", "medium.en"],
                   help="Modelo Whisper. Los '.en' son más precisos para inglés (solo con --backend whisper).")
    p.add_argument("--chunk-seconds", type=float, default=3.0,
                   help="Ventana de captura en segundos (default: 3).")
    p.add_argument("--max-history", type=int, default=50,
                   help="Máximo de líneas visibles por panel.")
    p.add_argument("--vad", action="store_true", default=True,
                   help="Filtro VAD de Silero dentro de Whisper (activo por defecto).")
    p.add_argument("--no-vad", dest="vad", action="store_false",
                   help="Desactiva el filtro VAD.")
    p.add_argument("--device", default=None,
                   help="Loopback: número o nombre parcial (--list-devices para ver).")
    p.add_argument("--mic", default=None,
                   help="Micrófono: número o nombre parcial.")
    p.add_argument("--prompt", default=None,
                   help="Prompt de contexto para Whisper (default: vocabulario médico).")
    args = p.parse_args()

    if args.list_devices:
        print_devices()
        return

    if args.setup_langs:
        sys.exit(0 if ensure_bidirectional() else 1)

    # Obtener API key de Deepgram desde entorno
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    if args.backend == "deepgram":
        if not deepgram_api_key:
            print(
                "ERROR: DEEPGRAM_API_KEY no encontrada.\n"
                "  Crea un archivo .env en la raíz del proyecto con:\n"
                "    DEEPGRAM_API_KEY=tu_api_key_aqui",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[Deepgram] API key cargada (últimos 4: …{deepgram_api_key[-4:]})", flush=True)
    else:
        # Backend Whisper: necesita Argos para traducción offline (fallback)
        if not ensure_bidirectional():
            print(
                "Ejecuta primero: python clinic_translate.py --setup-langs",
                file=sys.stderr,
            )
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
        backend=args.backend,
        deepgram_api_key=deepgram_api_key,
    )


if __name__ == "__main__":
    main()
