"""Módulo de traducción multi-motor.

Jerarquía configurable (según translation_engine):
  "deepl"   → DeepL API (mejor calidad EN/ES, gratis 500K chars/mes)
  "openai"  → GPT-4o-mini (contexto médico, centavos por consulta)
  "google"  → Google Translate vía deep_translator (gratis, sin API key)

Siempre con fallback a Argos Translate (offline) si todo falla.
"""
from __future__ import annotations

import os
import re
import sys

# Motor activo (se puede cambiar en runtime desde la UI)
_active_engine: str = "google"


def set_engine(engine: str) -> None:
    global _active_engine
    _active_engine = engine


def get_engine() -> str:
    return _active_engine


# ---------------------------------------------------------------------------
# Argos Translate (offline, último fallback)
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
# Validación de salida
# ---------------------------------------------------------------------------

_NON_LATIN_RE = re.compile(
    r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf'
    r'\u0400-\u04ff\u0600-\u06ff\uac00-\ud7af]',
)


def _is_garbage(result: str, source: str) -> bool:
    if not result:
        return False
    if "@@" in result or "###" in result:
        return True
    if len(source) > 0 and len(result) > max(200, len(source) * 5):
        return True
    if _NON_LATIN_RE.search(result):
        return True
    return False


# ---------------------------------------------------------------------------
# Motor 1: DeepL API (mejor calidad para EN/ES)
# ---------------------------------------------------------------------------

def _translate_deepl(text: str, from_lang: str, to_lang: str) -> str | None:
    """Traduce con DeepL API. Retorna None si no hay API key o falla."""
    api_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import urllib.request
        import urllib.parse
        import json

        # DeepL usa códigos especiales: "EN" y "ES" (mayúsculas)
        # Para target, EN necesita ser "EN-US" o "EN-GB"
        src = from_lang.upper()
        tgt = to_lang.upper()
        if tgt == "EN":
            tgt = "EN-US"

        # Free API usa api-free.deepl.com, Pro usa api.deepl.com
        host = "api-free.deepl.com" if api_key.endswith(":fx") else "api.deepl.com"
        url = f"https://{host}/v2/translate"

        data = urllib.parse.urlencode({
            "auth_key": api_key,
            "text": text,
            "source_lang": src,
            "target_lang": tgt,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())

        translated = result["translations"][0]["text"].strip()
        return translated if translated else None

    except Exception as e:
        print(f"[DeepL] Error ({from_lang}→{to_lang}): {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Motor 2: OpenAI GPT-4o-mini (traducción con contexto médico)
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM_PROMPT = (
    "You are a professional medical interpreter. Translate the following text "
    "accurately, preserving medical terminology and natural conversational tone. "
    "Only output the translation, nothing else."
)


def _translate_openai(text: str, from_lang: str, to_lang: str) -> str | None:
    """Traduce con GPT-4o-mini. Retorna None si no hay API key o falla."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    lang_names = {"en": "English", "es": "Spanish"}
    src_name = lang_names.get(from_lang, from_lang)
    tgt_name = lang_names.get(to_lang, to_lang)

    try:
        import urllib.request
        import json

        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n\n{text}"},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())

        translated = result["choices"][0]["message"]["content"].strip()
        # GPT a veces añade comillas o prefijos — limpiar
        translated = translated.strip('"').strip("'")
        return translated if translated else None

    except Exception as e:
        print(f"[OpenAI] Error ({from_lang}→{to_lang}): {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Motor 3: Google Translate vía deep_translator (gratis, sin API key)
# ---------------------------------------------------------------------------

def _translate_google(text: str, from_lang: str, to_lang: str) -> str | None:
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        result = GoogleTranslator(source=from_lang, target=to_lang).translate(text)
        if result and result.strip() and not _is_garbage(result, text):
            return result.strip()
        if result:
            print(f"[Google] Resultado descartado: {result[:60]!r}", flush=True)
    except Exception as e:
        print(f"[Google] Error ({from_lang}→{to_lang}): {e}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Función principal: translate_natural (usa el motor activo + fallbacks)
# ---------------------------------------------------------------------------

# Orden de fallback por motor
_ENGINE_CHAINS: dict[str, list] = {
    "deepl":  [_translate_deepl, _translate_google, _argos_translate],
    "openai": [_translate_openai, _translate_google, _argos_translate],
    "google": [_translate_google, _argos_translate],
}


def translate_natural(text: str, from_lang: str, to_lang: str) -> str:
    """Traduce usando el motor activo con fallback automático.

    Nunca lanza excepción. Devuelve "" si ningún motor pudo traducir.
    """
    text = (text or "").strip()
    if not text or from_lang == to_lang:
        return text

    chain = _ENGINE_CHAINS.get(_active_engine, _ENGINE_CHAINS["google"])

    for fn in chain:
        try:
            if fn == _argos_translate:
                result = fn(text, from_lang, to_lang)
            else:
                result = fn(text, from_lang, to_lang)
            if result and not _is_garbage(result, text):
                return result
        except Exception:
            continue

    return ""
