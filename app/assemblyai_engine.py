"""AssemblyAI Universal Streaming engine — WebSocket streaming.

Mantiene una conexión WebSocket persistente a AssemblyAI con el modelo
universal-streaming-multilingual (code-switching EN/ES/FR/DE/IT/PT).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlencode

import numpy as np


def _float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio.astype(np.float32), -1.0, 1.0)
    return (pcm * 32767).astype(np.int16).tobytes()


class AssemblyAIStreamer:
    """Conexión WebSocket streaming a AssemblyAI Universal Streaming.

    Uso:
        streamer = AssemblyAIStreamer(api_key, on_transcript=callback)
        streamer.start()
        streamer.send_audio(audio_chunk)
        streamer.stop()

    El callback recibe (text, lang) cuando end_of_turn=True.
    """

    BASE_URL = "wss://streaming.assemblyai.com/v3/ws"

    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str, str], None],
        on_error: Optional[Callable[[str], None]] = None,
        sample_rate: int = 16000,
    ) -> None:
        self._api_key = api_key
        self._on_transcript = on_transcript
        self._on_error = on_error or (lambda msg: print(f"[AssemblyAI] Error: {msg}", flush=True))
        self._sample_rate = sample_rate
        self._ws = None
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _build_url(self) -> str:
        params = {
            "sample_rate": self._sample_rate,
            "speech_model": "universal-streaming-multilingual",
            "language_detection": True,
        }
        return f"{self.BASE_URL}?{urlencode(params)}"

    def start(self) -> None:
        import websockets.sync.client  # type: ignore

        url = self._build_url()
        headers = {"Authorization": self._api_key}

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

        # Leer el mensaje Begin inicial
        try:
            begin_raw = self._ws.recv(timeout=10)
            begin_msg = json.loads(begin_raw)
            if begin_msg.get("type") == "Begin":
                print(f"[AssemblyAI] Conectado (session: {begin_msg.get('id', '?')})", flush=True)
        except Exception:
            print("[AssemblyAI] Conectado (sin Begin message)", flush=True)

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        print("[AssemblyAI] WebSocket cerrado", flush=True)

    def send_audio(self, audio: np.ndarray) -> None:
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

    def _recv_loop(self) -> None:
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

            if msg_type == "Turn":
                self._handle_turn(msg)
            elif msg_type == "Termination":
                print(f"[AssemblyAI] Sesión terminada ({msg.get('audio_duration_seconds', 0):.0f}s procesados)", flush=True)
                break

    def _handle_turn(self, msg: dict) -> None:
        end_of_turn = msg.get("end_of_turn", False)
        if not end_of_turn:
            return

        text = (msg.get("transcript") or "").strip()
        if not text:
            return

        lang = (msg.get("language_code") or "").split("-")[0].lower()
        if not lang:
            lang = "en"

        # Filtro mínimo de ruido
        t_lower = text.strip().lower().rstrip(".,!?")
        if t_lower in ("", ".", "uh", "um", "hmm", "ok", "okay"):
            return

        self._on_transcript(text, lang)
