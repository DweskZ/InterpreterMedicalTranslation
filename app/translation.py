"""Módulo de traducción: Google Translate (natural) con fallback a Argos (offline).

Jerarquía:
  1. deep_translator → GoogleTranslator (requiere internet, sin API key, más natural)
  2. argostranslate   → traducción offline (sin internet, más literal, fallback)
"""
from __future__ import annotations

import collections
import re
import sys


# ---------------------------------------------------------------------------
# Argos Translate (offline, fallback)
# ---------------------------------------------------------------------------

def _ensure_pair(from_code: str, to_code: str) -> bool:
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        print("Instala dependencias: pip install -r requirements.txt", file=sys.stderr)
        return False

    installed = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in installed if l.code == from_code), None)
    to_lang = next((l for l in installed if l.code == to_code), None)
    if from_lang and to_lang and from_lang.get_translation(to_lang):
        return True

    print(f"Descargando paquete Argos {from_code}->{to_code} (solo esta vez)...")
    argostranslate.package.update_package_index()
    pkg = next((p for p in argostranslate.package.get_available_packages()
                if p.from_code == from_code and p.to_code == to_code), None)
    if not pkg:
        print(f"No se encontró paquete Argos {from_code}->{to_code}.", file=sys.stderr)
        return False
    argostranslate.package.install_from_path(pkg.download())
    return True


def ensure_bidirectional() -> bool:
    return _ensure_pair("en", "es") and _ensure_pair("es", "en")


def _argos_translate(text: str, from_code: str, to_code: str) -> str:
    import argostranslate.translate
    text = (text or "").strip()
    return argostranslate.translate.translate(text, from_code, to_code) if text else ""


def en_to_es(text: str) -> str:
    return _argos_translate(text, "en", "es")


def es_to_en(text: str) -> str:
    return _argos_translate(text, "es", "en")


# ---------------------------------------------------------------------------
# Validación de salida de traducción
# ---------------------------------------------------------------------------

def _is_garbage(result: str, source: str) -> bool:
    """True si el resultado de la traducción parece basura.

    Detecta:
    - Artefactos de parseo de deep_translator (@@, ###, etc.)
    - Resultado mucho más largo que el texto original (>5x en caracteres)
    """
    if not result:
        return False

    # Artefactos conocidos de deep_translator al fallar
    if "@@" in result or "###" in result:
        return True

    # Resultado demasiado largo en relación al original (probable bucle)
    if len(source) > 0 and len(result) > max(200, len(source) * 5):
        return True

    return False


# ---------------------------------------------------------------------------
# Google Translate via deep_translator (natural, requiere internet)
# ---------------------------------------------------------------------------

def translate_natural(text: str, from_lang: str, to_lang: str) -> str:
    """Traduce usando Google Translate (natural). Cae a Argos si falla o sin internet.

    Args:
        text: texto a traducir.
        from_lang: código ISO 639-1 del idioma origen ("en", "es").
        to_lang:   código ISO 639-1 del idioma destino ("en", "es").

    Returns:
        Traducción como string. Nunca lanza excepción. Devuelve "" si no pudo
        producir una traducción válida.
    """
    text = (text or "").strip()
    if not text or from_lang == to_lang:
        return text

    # ── Intento 1: Google Translate vía deep_translator ──────────────────────
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        # Crear instancia fresca cada vez (evita estado compartido entre threads)
        result = GoogleTranslator(source=from_lang, target=to_lang).translate(text)
        if result and result.strip() and not _is_garbage(result, text):
            return result.strip()
        if result:
            print(f"[Traducción] Resultado descartado (garbage): {result[:60]!r}", flush=True)
    except Exception as e:
        print(f"[Traducción] Google falló ({from_lang}→{to_lang}): {e}", flush=True)

    # ── Intento 2: Argos Translate (offline) ─────────────────────────────────
    try:
        result = _argos_translate(text, from_lang, to_lang)
        if result and not _is_garbage(result, text):
            return result
    except Exception as e:
        print(f"[Traducción] Argos falló ({from_lang}→{to_lang}): {e}", flush=True)

    # Sin traducción válida: devolver vacío para que el panel no muestre basura
    return ""
