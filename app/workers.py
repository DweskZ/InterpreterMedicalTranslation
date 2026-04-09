"""Background worker threads for audio capture + transcription + translation."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import sys
from dataclasses import dataclass
from typing import Optional

from app.audio import AudioStream
from app import whisper_engine
from app import translation


@dataclass
class CaptionLine:
    source: str
    translated: str
    ts: float


def worker_system(
    out_q: "queue.Queue[Optional[CaptionLine]]",
    stop_evt: threading.Event,
    model,
    chunk_sec: float,
    vad_filter: bool,
    stream: AudioStream,
    prompt: str,
) -> None:
    """Loopback audio -> Whisper EN -> Argos EN->ES."""
    print(f"[Sistema] Capturando de: {stream.device['name']}", flush=True)

    while not stop_evt.is_set():
        try:
            raw = stream.read(chunk_sec, 16000)
        except Exception as e:
            out_q.put(CaptionLine(source="", translated=f"[Audio ERROR] {e}", ts=time.time()))
            print(traceback.format_exc(), file=sys.stderr)
            time.sleep(1.0)
            continue

        text_en = whisper_engine.transcribe(
            model, raw, language="en", vad_filter=vad_filter, prompt=prompt)

        if text_en:
            try:
                text_es = translation.en_to_es(text_en)
            except Exception as e:
                text_es = f"[Traduccion ERROR] {e}"
            out_q.put(CaptionLine(source=text_en, translated=text_es, ts=time.time()))
        else:
            out_q.put(CaptionLine(source="", translated="", ts=time.time()))


# --- Tab del interprete (mic ES->EN) - deshabilitado por ahora ---
#
# def worker_mic(
#     out_q: "queue.Queue[Optional[CaptionLine]]",
#     stop_evt: threading.Event,
#     model,
#     chunk_sec: float,
#     vad_filter: bool,
#     stream: AudioStream,
#     prompt: str,
# ) -> None:
#     """Microphone -> Whisper ES -> Argos ES->EN."""
#     print(f"[Mic] Capturando de: {stream.device['name']}", flush=True)
#
#     while not stop_evt.is_set():
#         try:
#             raw = stream.read(chunk_sec, 16000)
#         except Exception as e:
#             out_q.put(CaptionLine(source="", translated=f"[Audio ERROR] {e}", ts=time.time()))
#             print(traceback.format_exc(), file=sys.stderr)
#             time.sleep(1.0)
#             continue
#
#         text_es = whisper_engine.transcribe(
#             model, raw, language="es", vad_filter=vad_filter, prompt=prompt)
#
#         if text_es:
#             try:
#                 text_en = translation.es_to_en(text_es)
#             except Exception as e:
#                 text_en = f"[Translation ERROR] {e}"
#             out_q.put(CaptionLine(source=text_es, translated=text_en, ts=time.time()))
#         else:
#             out_q.put(CaptionLine(source="", translated="", ts=time.time()))
