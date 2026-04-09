"""Audio capture via WASAPI loopback (PyAudioWPatch).

Provides AudioStream for persistent callback-based recording and
device discovery helpers.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
def get_loopback_devices() -> list[dict]:
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    devices = []
    try:
        for lb in p.get_loopback_device_info_generator():
            devices.append(dict(lb))
    except Exception:
        pass
    p.terminate()
    return devices


def get_input_devices() -> list[dict]:
    """WASAPI microphones (physical input, not loopback)."""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    devices = []
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_idx = wasapi["index"]
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == wasapi_idx and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False):
                devices.append(dict(d))
    except Exception:
        pass
    p.terminate()
    return devices


def get_default_loopback() -> dict:
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_name = default_spk["name"]
        for lb in p.get_loopback_device_info_generator():
            if lb["name"].startswith(default_name[:25]):
                return dict(lb)
        for lb in p.get_loopback_device_info_generator():
            return dict(lb)
    finally:
        p.terminate()
    raise RuntimeError("No se encontro dispositivo loopback WASAPI.")


def get_default_microphone() -> dict:
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        return dict(p.get_device_info_by_index(wasapi["defaultInputDevice"]))
    finally:
        p.terminate()


def pick_device(devices: list[dict], hint: Optional[str], label: str) -> dict:
    if not devices:
        raise RuntimeError(f"No hay dispositivos {label} disponibles.")
    if hint is not None:
        if hint.isdigit():
            idx = int(hint)
            if 0 <= idx < len(devices):
                return devices[idx]
            raise RuntimeError(f"Indice {idx} fuera de rango (hay {len(devices)} {label}).")
        needle = hint.lower()
        for d in devices:
            if needle in d["name"].lower():
                return d
        raise RuntimeError(f"Ningun {label} contiene '{hint}'. Usa --list-devices.")
    return devices[0]


def pick_loopback(hint: Optional[str] = None) -> dict:
    if hint is None:
        return get_default_loopback()
    return pick_device(get_loopback_devices(), hint, "loopback")


def pick_microphone(hint: Optional[str] = None) -> dict:
    if hint is None:
        return get_default_microphone()
    return pick_device(get_input_devices(), hint, "microfono")


# ---------------------------------------------------------------------------
# Persistent audio stream (callback-based, shared PyAudio instance)
# ---------------------------------------------------------------------------
class AudioStream:
    """Continuously collects audio chunks via callback.

    All streams share one PyAudio instance because PyAudioWPatch crashes
    if multiple instances are created from different threads.
    """

    _pyaudio = None
    _pyaudio_lock = threading.Lock()

    @classmethod
    def _get_pyaudio(cls):
        with cls._pyaudio_lock:
            if cls._pyaudio is None:
                import pyaudiowpatch as pyaudio
                cls._pyaudio = pyaudio.PyAudio()
            return cls._pyaudio

    @classmethod
    def shutdown(cls):
        with cls._pyaudio_lock:
            if cls._pyaudio is not None:
                try:
                    cls._pyaudio.terminate()
                except Exception:
                    pass
                cls._pyaudio = None

    def __init__(self, device: dict):
        import pyaudiowpatch as pyaudio
        self.device = device
        self.channels = device["maxInputChannels"]
        self.dev_rate = int(device["defaultSampleRate"])
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

        p = self._get_pyaudio()
        self._stream = p.open(
            format=pyaudio.paFloat32, channels=self.channels,
            rate=self.dev_rate, input=True,
            input_device_index=device["index"],
            frames_per_buffer=512, stream_callback=self._cb,
        )
        self._stream.start_stream()

    def _cb(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        with self._lock:
            self._chunks.append(np.frombuffer(in_data, dtype=np.float32).copy())
        return (None, pyaudio.paContinue)

    def read(self, seconds: float, target_rate: int) -> np.ndarray:
        """Sleep for *seconds*, drain accumulated chunks -> mono float32 at *target_rate*."""
        time.sleep(seconds)
        with self._lock:
            chunks = self._chunks.copy()
            self._chunks.clear()

        if not chunks:
            return np.zeros(int(target_rate * seconds), dtype=np.float32)

        audio = np.concatenate(chunks)
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels).mean(axis=1)

        if self.dev_rate != target_rate:
            from scipy import signal
            num = int(len(audio) * target_rate / self.dev_rate)
            if num > 0:
                audio = signal.resample(audio.astype(np.float64), num).astype(np.float32)

        return audio.astype(np.float32)

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def print_devices() -> None:
    loopbacks = get_loopback_devices()
    mics = get_input_devices()

    default_lb = ""
    try:
        default_lb = get_default_loopback()["name"]
    except Exception:
        pass
    default_mic = ""
    try:
        default_mic = get_default_microphone()["name"]
    except Exception:
        pass

    print("\n=== LOOPBACK (audio del sistema, --device) ===")
    print(f"{'#':>3}  {'Nombre':55s}  {'Ch':>3}  {'Rate':>6}")
    print("-" * 75)
    for i, d in enumerate(loopbacks):
        m = "  <-- ACTIVO" if d["name"] == default_lb else ""
        print(f"{i:>3}  {d['name']:55s}  {d['maxInputChannels']:>3}  {int(d['defaultSampleRate']):>6}{m}")

    print("\n=== MICROFONOS (entrada fisica, --mic) ===")
    print(f"{'#':>3}  {'Nombre':55s}  {'Ch':>3}  {'Rate':>6}")
    print("-" * 75)
    for i, d in enumerate(mics):
        m = "  <-- ACTIVO" if d["name"] == default_mic else ""
        print(f"{i:>3}  {d['name']:55s}  {d['maxInputChannels']:>3}  {int(d['defaultSampleRate']):>6}{m}")
    print()
