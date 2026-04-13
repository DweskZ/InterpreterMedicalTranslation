"""Background worker threads for audio capture + transcription + translation."""
from __future__ import annotations

import collections
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
      - Sin contexto aún  -> MEDICAL_PROMPT completo   (~200 tokens)
      - Con contexto       -> MEDICAL_PROMPT_SHORT (~30 tokens)
                           + últimas MAX_WORDS palabras (~130 tokens)
                           = ~160 tokens  (holgura para evitar truncado)

    Salvaguardas:
      - Solo entra texto que pasó filtros de alucinación y repetición.
      - Anti-loop: rechaza texto idéntico al segmento anterior (evita
        retroalimentar frases repetitivas al decoder).
      - Reset automático tras RESET_SECS de silencio.
    """

    MAX_WORDS: int = 100
    RESET_SECS: float = 45.0
    _RECENT_DEDUP: int = 5  # cuántos segmentos recientes recordar para dedup

    def __init__(self) -> None:
        self._words: list[str] = []
        self._last_ts: float = 0.0
        self._recent: collections.deque[str] = collections.deque(maxlen=self._RECENT_DEDUP)

    def update(self, text: str) -> None:
        """Agrega texto al contexto si no es duplicado reciente."""
        now = time.monotonic()
        if self._last_ts and (now - self._last_ts) > self.RESET_SECS:
            self._words.clear()
            self._recent.clear()
        self._last_ts = now

        normalized = text.strip().lower()
        if normalized in self._recent:
            return
        self._recent.append(normalized)

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
        self._recent.clear()
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
_ES_FALLBACK_THRESHOLD: float = 0.8  # si confianza EN < 80%, probar ES
_DEDUP_HISTORY: int = 3              # líneas recientes para deduplicar


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
    """Loopback audio -> Whisper (EN/ES doble pasada) -> traducción bidireccional."""
    print(f"[Whisper] Capturando de: {stream.device['name']}", flush=True)

    ctx = _RollingContext()
    recent_lines: collections.deque[str] = collections.deque(maxlen=_DEDUP_HISTORY)

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

        # Auto-detección: Whisper decide si es EN o ES.
        # IMPORTANTE: los modelos ".en" NO pueden detectar español.
        # Usar modelos sin ".en" (base, small, medium) para bilingüe.
        text, detected_lang, prob = whisper_engine.transcribe(
            model_holder.model, raw, language=None,
            vad_filter=vad_filter, prompt=effective_prompt,
        )
        lang = detected_lang if detected_lang in ("en", "es") else "en"

        if text:
            # Deduplicar líneas consecutivas idénticas
            norm_text = text.strip().lower()
            if norm_text in recent_lines:
                continue
            recent_lines.append(norm_text)

            ctx.update(text)
            lang = lang if lang in ("en", "es") else "en"
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


# ---------------------------------------------------------------------------
# Worker AssemblyAI (cloud) - Universal Streaming Multilingual
# ---------------------------------------------------------------------------
_AAI_CHUNK_SEC: float = 0.1

def worker_assemblyai(
    out_q: "queue.Queue[Optional[CaptionLine]]",
    stop_evt: threading.Event,
    api_key: str,
    chunk_sec: float,
    stream: AudioStream,
    normalize_evt: threading.Event,
) -> None:
    """Loopback audio -> AssemblyAI Universal Streaming (code-switching EN/ES)."""
    from app.assemblyai_engine import AssemblyAIStreamer

    dev_rate = stream.dev_rate
    print(f"[AssemblyAI] Capturando de: {stream.device['name']} ({dev_rate} Hz)", flush=True)

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
        out_q.put(CaptionLine(source="", translated=f"[AssemblyAI] {msg}",
                               ts=time.time(), source_lang="en"))
        print(f"[AssemblyAI] {msg}", file=sys.stderr, flush=True)

    streamer: Optional[AssemblyAIStreamer] = None
    try:
        streamer = AssemblyAIStreamer(
            api_key,
            on_transcript=_on_transcript,
            on_error=_on_error,
            sample_rate=dev_rate,
        )
        streamer.start()
    except Exception as e:
        out_q.put(CaptionLine(source="", translated=f"[AssemblyAI ERROR] {e}",
                               ts=time.time(), source_lang="en"))
        print(f"[AssemblyAI] Fallo al conectar: {e}", file=sys.stderr, flush=True)
        return

    try:
        while not stop_evt.is_set():
            try:
                raw = stream.read(_AAI_CHUNK_SEC, dev_rate)
            except Exception as e:
                _on_error(f"[Audio ERROR] {e}")
                print(traceback.format_exc(), file=sys.stderr)
                time.sleep(1.0)
                continue

            if normalize_evt.is_set():
                raw = _normalize_rms(raw)

            streamer.send_audio(raw)
    finally:
        if streamer:
            streamer.stop()
