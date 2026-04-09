"""Whisper transcription engine (faster-whisper with CUDA)."""
from __future__ import annotations

import threading

import numpy as np

_lock = threading.Lock()

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


def transcribe(model, audio: np.ndarray, language: str, *,
               vad_filter: bool, prompt: str = "") -> str:
    """Transcribe an audio chunk (thread-safe via lock)."""
    if audio.size < 8000:
        return ""
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    kwargs: dict = dict(
        language=language, task="transcribe", beam_size=1,
        vad_filter=vad_filter, without_timestamps=True,
    )
    if prompt:
        kwargs["initial_prompt"] = prompt
    with _lock:
        segs, _ = model.transcribe(audio, **kwargs)
        return " ".join(s.text.strip() for s in segs if s.text).strip()
