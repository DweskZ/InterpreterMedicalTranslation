"""Background worker threads for audio capture + transcription + translation."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.audio import AudioStream
from app import whisper_engine
from app import translation

# ---------------------------------------------------------------------------
# Audio processing constants
# ---------------------------------------------------------------------------
_SILENCE_RMS: float = 0.01   # chunks por debajo de este RMS se descartan
_TARGET_RMS: float = 0.12    # nivel objetivo para normalización
_MAX_GAIN: float = 15.0      # ganancia máxima permitida (evita amplificar ruido)


def _is_silent(audio: np.ndarray) -> bool:
    """True si el chunk de audio no supera el umbral de energía mínimo."""
    if audio.size == 0:
        return True
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < _SILENCE_RMS


def _normalize_rms(audio: np.ndarray) -> np.ndarray:
    """Amplifica el chunk para que su energía RMS alcance _TARGET_RMS."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < _SILENCE_RMS:
        return audio
    gain = min(_TARGET_RMS / rms, _MAX_GAIN)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling context (solo Whisper)
# ---------------------------------------------------------------------------
class _RollingContext:
    """Ventana deslizante de palabras recientes para usar como initial_prompt.

    Estrategia de tokens (~224 límite de Whisper):
      - Sin contexto aún  → MEDICAL_PROMPT completo   (~200 tokens)
      - Con contexto       → MEDICAL_PROMPT_SHORT (~30 tokens)
                           + últimas MAX_WORDS palabras (~130 tokens)
                           = ~160 tokens  (holgura para evitar truncado)

    Salvaguardas contra regresión:
      - Solo entra texto que pasó el filtro de alucinaciones (garantizado por
        whisper_engine.transcribe).
      - Reset automático tras RESET_SECS de silencio (conversación nueva).
      - Capa máxima de palabras para no saturar el contexto del decoder.
    """

    MAX_WORDS: int = 100        # ~130 tokens, deja margen con el header corto
    RESET_SECS: float = 45.0   # silencio prolongado = probable cambio de tema

    def __init__(self) -> None:
        self._words: list[str] = []
        self._last_ts: float = 0.0

    def update(self, text: str) -> None:
        """Agrega texto al contexto; resetea si hubo silencio prolongado."""
        now = time.monotonic()
        if self._last_ts and (now - self._last_ts) > self.RESET_SECS:
            self._words.clear()
        self._last_ts = now
        self._words.extend(text.split())
        if len(self._words) > self.MAX_WORDS:
            self._words = self._words[-self.MAX_WORDS:]

    def build_prompt(self, full_prompt: str, short_prompt: str) -> str:
        """Devuelve el prompt óptimo según el estado del contexto."""
        if not self._words:
            return full_prompt
        return f"{short_prompt} {' '.join(self._words)}"

    def reset(self) -> None:
        self._words.clear()
        self._last_ts = 0.0


@dataclass
class CaptionLine:
    source: str
    translated: str
    ts: float
    source_lang: str = field(default="en")   # "en" o "es"


# ---------------------------------------------------------------------------
# Worker Whisper (local)
# ---------------------------------------------------------------------------
def worker_system(
    out_q: "queue.Queue[Optional[CaptionLine]]",
    stop_evt: threading.Event,
    model_holder: "whisper_engine.ModelHolder",
    chunk_sec: float,
    vad_filter: bool,
    stream: AudioStream,
    prompt: str,
    normalize_evt: threading.Event,
) -> None:
    """Loopback audio -> Whisper (auto-detecta EN/ES) -> traducción natural bidireccional."""
    print(f"[Whisper] Capturando de: {stream.device['name']}", flush=True)

    ctx = _RollingContext()

    while not stop_evt.is_set():
        try:
            raw = stream.read(chunk_sec, 16000)
        except Exception as e:
            out_q.put(CaptionLine(source="", translated=f"[Audio ERROR] {e}",
                                  ts=time.time(), source_lang="en"))
            print(traceback.format_exc(), file=sys.stderr)
            time.sleep(1.0)
            continue

        if _is_silent(raw):
            continue

        if normalize_evt.is_set():
            raw = _normalize_rms(raw)

        effective_prompt = ctx.build_prompt(prompt, whisper_engine.MEDICAL_PROMPT_SHORT)

        # language=None → Whisper auto-detecta EN o ES
        text, lang = whisper_engine.transcribe(
            model_holder.model, raw, language=None,
            vad_filter=vad_filter, prompt=effective_prompt,
        )

        if text:
            ctx.update(text)
            lang = lang or "en"
            to_lang = "en" if lang == "es" else "es"
            try:
                translated = translation.translate_natural(text, lang, to_lang)
            except Exception as e:
                translated = f"[Traduccion ERROR] {e}"
            out_q.put(CaptionLine(source=text, translated=translated,
                                  ts=time.time(), source_lang=lang))
        else:
            out_q.put(CaptionLine(source="", translated="", ts=time.time(), source_lang="en"))


# ---------------------------------------------------------------------------
# Worker Deepgram (cloud) - WebSocket streaming con Nova-3
# ---------------------------------------------------------------------------
_DG_CHUNK_SEC: float = 0.1   # enviar audio cada 100ms para flujo continuo

def worker_deepgram(
    out_q: "queue.Queue[Optional[CaptionLine]]",
    stop_evt: threading.Event,
    api_key: str,
    chunk_sec: float,
    stream: AudioStream,
    normalize_evt: threading.Event,
) -> None:
    """Loopback audio -> Deepgram Nova-3 (WebSocket streaming, code-switching EN/ES).

    A diferencia de Whisper que acumula 3s de audio y procesa de golpe,
    aquí enviamos audio cada 100ms como un flujo continuo. Deepgram
    mantiene contexto internamente y devuelve frases completas.
    """
    from app.deepgram_engine import DeepgramStreamer

    dev_rate = stream.dev_rate
    print(f"[Deepgram] Capturando de: {stream.device['name']} ({dev_rate} Hz)", flush=True)

    def _on_transcript(text: str, lang: str) -> None:
        lang = lang or "en"
        to_lang = "en" if lang == "es" else "es"
        try:
            translated = translation.translate_natural(text, lang, to_lang)
        except Exception as e:
            translated = f"[Traduccion ERROR] {e}"
        out_q.put(CaptionLine(source=text, translated=translated,
                               ts=time.time(), source_lang=lang))

    def _on_error(msg: str) -> None:
        out_q.put(CaptionLine(source="", translated=f"[Deepgram] {msg}",
                               ts=time.time(), source_lang="en"))
        print(f"[Deepgram] {msg}", file=sys.stderr, flush=True)

    streamer: Optional[DeepgramStreamer] = None
    try:
        streamer = DeepgramStreamer(
            api_key,
            on_transcript=_on_transcript,
            on_error=_on_error,
            sample_rate=dev_rate,
        )
        streamer.start()
    except Exception as e:
        out_q.put(CaptionLine(source="", translated=f"[Deepgram ERROR] {e}",
                               ts=time.time(), source_lang="en"))
        print(f"[Deepgram] Fallo al conectar: {e}", file=sys.stderr, flush=True)
        return

    try:
        while not stop_evt.is_set():
            try:
                raw = stream.read(_DG_CHUNK_SEC, dev_rate)
            except Exception as e:
                _on_error(f"[Audio ERROR] {e}")
                print(traceback.format_exc(), file=sys.stderr)
                time.sleep(1.0)
                continue

            if normalize_evt.is_set():
                raw = _normalize_rms(raw)

            # Enviar TODO el audio (incluyendo silencio) para que Deepgram
            # mantenga el contexto temporal y detecte correctamente
            # el inicio y fin de cada utterance.
            streamer.send_audio(raw)
    finally:
        if streamer:
            streamer.stop()
