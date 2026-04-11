"""Pre-carga DLLs de CUDA antes de importar faster_whisper / ctranslate2.

DEBE importarse antes que cualquier otro modulo que use CUDA.

Soporta dos entornos:
  - Desarrollo normal: busca en site-packages de Python
  - Ejecutable PyInstaller (frozen): busca junto al .exe
"""
from __future__ import annotations

import sys


def preload() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    import os
    from pathlib import Path

    path_dirs: list[str] = []
    seen: set[str] = set()

    def reg(d: Path) -> None:
        k = str(d.resolve())
        if k in seen or not d.is_dir():
            return
        seen.add(k)
        path_dirs.append(k)
        try:
            os.add_dll_directory(k)
        except (OSError, AttributeError, ValueError):
            pass

    # --- Entorno frozen (PyInstaller --onedir) ---
    # Las DLLs se colocan junto al .exe durante el build.
    if getattr(sys, "frozen", False):
        reg(Path(sys.executable).parent)

    # --- Entorno de desarrollo: site-packages ---
    else:
        import site

        roots: list[str] = []
        try:
            u = site.getusersitepackages()
            if u:
                roots.append(u)
        except Exception:
            pass
        try:
            roots.extend(x for x in site.getsitepackages() if x)
        except Exception:
            pass

        for root in roots:
            nv = Path(root) / "nvidia"
            if not nv.is_dir():
                continue
            for pkg in nv.iterdir():
                if pkg.is_dir():
                    for sub in ("bin", "lib"):
                        reg(pkg / sub)

        for env in ("CUDA_PATH", "CUDA_HOME"):
            v = os.environ.get(env)
            if v:
                for sub in ("bin", "lib", r"lib\x64"):
                    reg(Path(v) / sub)

    if path_dirs:
        os.environ["PATH"] = os.pathsep.join(path_dirs) + os.pathsep + os.environ.get("PATH", "")

    # Pre-carga explícita de las DLLs críticas para ctranslate2
    for dll in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
        for d in path_dirs:
            candidate = Path(d) / dll
            if candidate.exists():
                try:
                    ctypes.WinDLL(str(candidate))
                except OSError:
                    pass
                break


preload()
