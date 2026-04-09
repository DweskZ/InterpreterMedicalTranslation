"""Argos Translate wrappers (offline EN<->ES)."""
from __future__ import annotations

import sys


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
        print(f"No se encontro paquete Argos {from_code}->{to_code}.", file=sys.stderr)
        return False
    argostranslate.package.install_from_path(pkg.download())
    return True


def ensure_bidirectional() -> bool:
    return _ensure_pair("en", "es") and _ensure_pair("es", "en")


def en_to_es(text: str) -> str:
    import argostranslate.translate
    text = (text or "").strip()
    return argostranslate.translate.translate(text, "en", "es") if text else ""


def es_to_en(text: str) -> str:
    import argostranslate.translate
    text = (text or "").strip()
    return argostranslate.translate.translate(text, "es", "en") if text else ""
