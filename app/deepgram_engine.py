"""Deepgram cloud transcription engine — WebSocket streaming.

Mantiene una conexión WebSocket persistente a Deepgram Nova-3 con
language=multi (code-switching EN/ES). El audio se envía como flujo
continuo, igual que una llamada telefónica, y los resultados llegan
en tiempo real con contexto completo.
"""
from __future__ import annotations

import collections
import json
import re
import struct
import threading
import time
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constantes de filtrado (mismas que antes)
# ---------------------------------------------------------------------------
_NOISE_PHRASES: frozenset[str] = frozenset({
    "", ".", "...", "okay", "ok", "uh", "um", "hmm", "hm",
    "thank you", "thanks", "bye", "goodbye",
    "you", "i", "the", "a",
})
_MAX_REPEAT_RATIO: float = 0.40
_MIN_WORDS_FOR_REPEAT_CHECK: int = 2
_MIN_CONFIDENCE: float = 0.45


def _is_noise(text: str) -> bool:
    t = text.strip().lower().rstrip(".,!?")
    return t in _NOISE_PHRASES or len(t.replace(" ", "")) <= 1


def _is_repetitive(text: str) -> bool:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    if len(words) < _MIN_WORDS_FOR_REPEAT_CHECK:
        return False
    counts = collections.Counter(words)
    _, top_count = counts.most_common(1)[0]
    if top_count / len(words) > _MAX_REPEAT_RATIO:
        return True
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if bigrams:
        bi_counts = collections.Counter(bigrams)
        _, bi_top = bi_counts.most_common(1)[0]
        if bi_top / len(bigrams) > _MAX_REPEAT_RATIO:
            return True
    return False


def _dominant_language(words: list[dict]) -> str:
    """Idioma con más palabras en el resultado."""
    if not words:
        return ""
    lang_counts: dict[str, int] = {}
    for w in words:
        lang = (w.get("language") or "").split("-")[0].lower()
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    if not lang_counts:
        return ""
    return max(lang_counts, key=lang_counts.get)  # type: ignore[arg-type]


def _float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convierte float32 [-1, 1] a PCM int16 little-endian bytes."""
    pcm = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (pcm * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Clase principal: conexión WebSocket persistente
# ---------------------------------------------------------------------------
class DeepgramStreamer:
    """Conexión WebSocket streaming a Deepgram Nova-3.

    Uso típico:
        streamer = DeepgramStreamer(api_key, on_transcript=callback)
        streamer.start()
        ...
        streamer.send_audio(audio_chunk)   # llamar repetidamente
        ...
        streamer.stop()

    El callback on_transcript recibe (text: str, lang: str) cada vez que
    Deepgram produce un resultado final (is_final=True con speech_final=True).
    """

    WS_URL = "wss://api.deepgram.com/v1/listen"
    KEEPALIVE_INTERVAL = 8.0   # segundos entre KeepAlive messages

    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str, str], None],
        on_error: Optional[Callable[[str], None]] = None,
        sample_rate: int = 16000,
    ) -> None:
        self._api_key = api_key
        self._on_transcript = on_transcript
        self._on_error = on_error or (lambda msg: print(f"[Deepgram] Error: {msg}", flush=True))
        self._sample_rate = sample_rate
        self._ws = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _build_url(self) -> str:
        params = (
            f"model=nova-3"
            f"&language=multi"
            f"&punctuate=true"
            f"&smart_format=true"
            f"&encoding=linear16"
            f"&sample_rate={self._sample_rate}"
            f"&channels=1"
            f"&interim_results=false"
        )
        return f"{self.WS_URL}?{params}"

    def start(self) -> None:
        """Abre la conexión WebSocket e inicia los hilos de recepción y keepalive."""
        import websockets.sync.client  # type: ignore

        url = self._build_url()
        headers = {"Authorization": f"Token {self._api_key}"}

        try:
            self._ws = websockets.sync.client.connect(
                url,
                additional_headers=headers,
                open_timeout=10,
                close_timeout=5,
            )
        except Exception as exc:
            self._on_error(f"No se pudo conectar: {exc}")
            raise

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()
        print("[Deepgram] WebSocket conectado (Nova-3, streaming)", flush=True)

    def stop(self) -> None:
        """Cierra la conexión limpiamente enviando CloseStream."""
        self._running = False
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        print("[Deepgram] WebSocket cerrado", flush=True)

    def send_audio(self, audio: np.ndarray) -> None:
        """Envía un chunk de audio float32 por el WebSocket."""
        if not self._running or not self._ws:
            return
        pcm_bytes = _float32_to_int16_bytes(audio)
        if not pcm_bytes:
            return
        try:
            with self._lock:
                self._ws.send(pcm_bytes)
        except Exception as exc:
            self._on_error(f"Error enviando audio: {exc}")

    # ── Hilos internos ────────────────────────────────────────────────────

    def _recv_loop(self) -> None:
        """Hilo que escucha mensajes del WebSocket y procesa transcripciones."""
        while self._running and self._ws:
            try:
                raw = self._ws.recv(timeout=5)
            except TimeoutError:
                continue
            except Exception as exc:
                if self._running:
                    self._on_error(f"Conexión perdida: {exc}")
                break

            if isinstance(raw, bytes):
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "Results":
                self._handle_result(msg)
            elif msg_type == "Error":
                self._on_error(msg.get("description", str(msg)))

    def _keepalive_loop(self) -> None:
        """Envía KeepAlive periódicamente para mantener la conexión."""
        while self._running and self._ws:
            time.sleep(self.KEEPALIVE_INTERVAL)
            if not self._running or not self._ws:
                break
            try:
                with self._lock:
                    self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                break

    def _handle_result(self, msg: dict) -> None:
        """Procesa un mensaje de tipo Results de Deepgram.

        Emite cada fragmento is_final=True directamente como una frase.
        Deepgram ya se encarga de dar contexto al audio completo — cada
        is_final es una unidad coherente de transcripción.
        """
        is_final = msg.get("is_final", False)

        if not is_final:
            return

        try:
            alt = msg["channel"]["alternatives"][0]
            text = alt.get("transcript", "").strip()
            confidence = float(alt.get("confidence", 1.0))
            words = alt.get("words", [])
        except (KeyError, IndexError):
            return

        if not text:
            return

        lang = _dominant_language(words)
        if not lang:
            langs_list = alt.get("languages", [])
            if langs_list:
                lang = str(langs_list[0]).split("-")[0].lower()

        if _is_noise(text):
            return

        if confidence < _MIN_CONFIDENCE:
            print(f"[Deepgram] Baja confianza ({confidence:.2f}), descartando: {text[:60]!r}", flush=True)
            return

        if _is_repetitive(text):
            print(f"[Deepgram] Repetición descartada: {text[:60]!r}", flush=True)
            return

        self._on_transcript(text, lang)
