"""Whisper transcription engine (faster-whisper with CUDA)."""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional

import numpy as np

_lock = threading.Lock()


class ModelHolder:
    """Contenedor thread-safe del modelo Whisper activo.

    Permite cambiar de modelo en caliente sin reiniciar la app.
    El worker siempre lee `holder.model` justo antes de cada transcripción.
    """

    def __init__(self, model, size: str) -> None:
        self._model = model
        self._size = size
        self._rw_lock = threading.Lock()

    @property
    def model(self):
        with self._rw_lock:
            return self._model

    @property
    def size(self) -> str:
        with self._rw_lock:
            return self._size

    def swap(
        self,
        new_size: str,
        on_start: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        """Carga un nuevo modelo en background y lo reemplaza atómicamente."""

        def _bg() -> None:
            if on_start:
                on_start(new_size)
            try:
                new_model = load(new_size)
                with self._rw_lock:
                    self._model = new_model
                    self._size = new_size
                if on_done:
                    on_done(new_size)
            except Exception as exc:  # noqa: BLE001
                if on_error:
                    on_error(new_size, exc)

        threading.Thread(target=_bg, daemon=True).start()


MEDICAL_PROMPT = (
    "Medical interpreter in a hospital. Emergency room, surgery, pediatrics, cardiology, "
    "neurology, orthopedics, obstetrics, gynecology, radiology, oncology, psychiatry, "
    "dermatology, ophthalmology, urology, endocrinology, gastroenterology, pulmonology, "
    "nephrology, hematology, rheumatology, infectious diseases, anesthesiology. "
    "Patient intake, triage, vital signs, blood pressure, heart rate, oxygen saturation, "
    "temperature, pain scale, weight, height, BMI, allergies, medical history. "
    "Symptoms: chest pain, shortness of breath, abdominal pain, headache, dizziness, "
    "nausea, vomiting, diarrhea, fever, chills, bleeding, fracture, laceration, swelling, "
    "rash, numbness, tingling, seizure, fainting, allergic reaction, difficulty breathing. "
    "Medications, dosage, milligrams, prescriptions, refills, side effects, IV, injection, "
    "antibiotic, ibuprofen, acetaminophen, insulin, epinephrine, morphine, anesthesia. "
    "Procedures: blood work, CBC, CT scan, MRI, X-ray, ultrasound, EKG, ECG, biopsy, "
    "endoscopy, colonoscopy, catheter, intubation, ventilator, dialysis, transfusion, "
    "stitches, sutures, splint, cast, physical therapy. "
    "Baby, infant, toddler, child, pregnancy, contractions, delivery, C-section, epidural, "
    "NICU, breastfeeding, vaccination, immunization. "
    "Insurance, copay, deductible, prior authorization, consent forms, HIPAA, "
    "discharge instructions, follow-up appointment, referral, specialist."
)

# Versión corta (~25 tokens) usada como anclaje de dominio cuando el rolling
# context ya ocupa el resto del espacio disponible en initial_prompt (~224 tokens).
MEDICAL_PROMPT_SHORT = (
    "Medical interpreter. Hospital: emergency, surgery, cardiology, pediatrics, neurology. "
    "Symptoms, medications, diagnosis, procedures, vital signs, allergies, prescriptions."
)


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------
# Whisper genera estas frases cuando recibe silencio o audio de muy baja
# energía. Son patrones documentados y reproducibles del modelo.
_HALLUCINATION_EXACT: frozenset[str] = frozenset({
    "you", "thank you", "thanks", "thanks for watching", "thanks for listening",
    "bye", "goodbye", "ok", "okay", "uh", "um", "hmm", "hm",
    "please subscribe", "like and subscribe", "subscribe",
    "subtitles by", "translated by", "transcribed by",
    "captions by", "closed captions",
    "this video is brought to you by", "brought to you by",
    "music", "silence",
})

_MUSIC_CHARS: frozenset[str] = frozenset("♪♫♬♩")


def _is_hallucination(text: str) -> bool:
    """True si el texto es una alucinación típica de Whisper en silencio."""
    if not text:
        return False
    t = text.strip()
    # Notación musical pura
    if all(c in _MUSIC_CHARS or c.isspace() or c in ",.!?-[]" for c in t):
        return True
    # Normalizar y comparar contra lista negra
    t_norm = re.sub(r"[^\w\s]", "", t.lower()).strip()
    if t_norm in _HALLUCINATION_EXACT:
        return True
    # Texto de 1-2 caracteres reales casi siempre es ruido
    if len(t_norm.replace(" ", "")) <= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _smoke_test(model) -> None:
    segs, _ = model.transcribe(
        np.zeros(32000, dtype=np.float32),
        language="en", beam_size=1, vad_filter=False, without_timestamps=True,
    )
    list(segs)


def load(model_size: str):
    """Load a Whisper model, trying CUDA first then CPU."""
    from faster_whisper import WhisperModel

    for dev, ct in (("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")):
        try:
            m = WhisperModel(model_size, device=dev, compute_type=ct)
            _smoke_test(m)
            print(f"Whisper cargado: modelo={model_size} device={dev} compute_type={ct}", flush=True)
            return m
        except Exception as e:
            if dev == "cuda":
                print(f"CUDA fallo: {e!r} -> probando siguiente...", flush=True)
                continue
            raise
    raise RuntimeError("No se pudo cargar Whisper en ninguna configuracion.")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe(
    model,
    audio: np.ndarray,
    language: "str | None",
    *,
    vad_filter: bool,
    prompt: str = "",
) -> tuple[str, str]:
    """Transcribe an audio chunk (thread-safe via lock).

    Args:
        language: código ISO 639-1 ("en", "es") o None para auto-detección.

    Returns:
        (texto, idioma_detectado)  — idioma_detectado es "" si no se pudo determinar.
    """
    if audio.size < 8000:
        return "", ""
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    kwargs: dict = {
        "task": "transcribe", "beam_size": 1,
        "vad_filter": vad_filter, "without_timestamps": True,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    with _lock:
        segs, info = model.transcribe(audio, **kwargs)
        result = " ".join(s.text.strip() for s in segs if s.text).strip()
    detected = (info.language or "").split("-")[0].lower() if hasattr(info, "language") else ""
    if _is_hallucination(result):
        return "", ""
    return result, detected
