"""Tkinter overlay UI with split panels and tabbed interface."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from app.audio import AudioStream, pick_loopback
from app.workers import CaptionLine, worker_system, worker_deepgram
from app import whisper_engine

MODEL_CHOICES = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"]

FONT_FAMILY = "Segoe UI"


def run(
    model_size: str,
    chunk_sec: float,
    max_lines: int,
    vad_filter: bool,
    device_hint: Optional[str],
    mic_hint: Optional[str],
    prompt: str,
    backend: str = "whisper",
    deepgram_api_key: str = "",
) -> None:
    import tkinter as tk
    from tkinter import font as tkfont

    q_sys: queue.Queue[Optional[CaptionLine]] = queue.Queue(maxsize=64)
    stop_evt = threading.Event()
    effective_model = model_size

    loopback_dev = pick_loopback(device_hint)
    stream_sys = AudioStream(loopback_dev)
    print(f"Dispositivo: {loopback_dev['name']}", flush=True)

    root = tk.Tk()
    root.title("Clinic Translate - Medical Interpreter")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.93)
    root.configure(bg="#1a1a1a")
    root.geometry("1100x540+40+40")
    root.minsize(700, 340)

    header_font = tkfont.Font(family=FONT_FAMILY, size=14, weight="bold")
    text_font   = tkfont.Font(family=FONT_FAMILY, size=13)
    ts_font     = tkfont.Font(family="Consolas", size=10)
    small_font  = tkfont.Font(family=FONT_FAMILY, size=10)
    badge_font  = tkfont.Font(family="Consolas", size=9, weight="bold")

    # ── Toolbar ──────────────────────────────────────────────────────────────
    toolbar = tk.Frame(root, bg="#111111", pady=4)
    toolbar.pack(fill="x", side="top")

    # Selector de backend
    tk.Label(toolbar, text="Backend:", fg="#888888", bg="#111111",
             font=small_font).pack(side="left", padx=(10, 4))

    backend_var = tk.StringVar(value=backend)
    backend_menu = tk.OptionMenu(toolbar, backend_var, "whisper", "deepgram")
    backend_menu.config(
        bg="#2a2a2a", fg="#dddddd", activebackground="#3a3a3a", activeforeground="#ffffff",
        font=small_font, bd=0, highlightthickness=1, highlightbackground="#444444",
        relief="flat", padx=6,
    )
    backend_menu["menu"].config(
        bg="#2a2a2a", fg="#dddddd", activebackground="#4a9a7a", activeforeground="#ffffff",
        font=small_font,
    )
    backend_menu.pack(side="left")

    backend_badge_var = tk.StringVar(value="")
    backend_badge = tk.Label(toolbar, textvariable=backend_badge_var,
                             fg="#4a9a7a", bg="#111111", font=badge_font)
    backend_badge.pack(side="left", padx=(4, 0))

    tk.Frame(toolbar, bg="#333333", width=1).pack(side="left", fill="y", padx=(10, 8))

    # Selector de modelo Whisper
    whisper_label = tk.Label(toolbar, text="Modelo:", fg="#888888", bg="#111111",
                             font=small_font)
    whisper_label.pack(side="left", padx=(0, 4))

    model_var = tk.StringVar(value=effective_model)
    model_status_var = tk.StringVar(value=f"⟳  {effective_model}")

    model_menu = tk.OptionMenu(toolbar, model_var, *MODEL_CHOICES)
    model_menu.config(
        bg="#2a2a2a", fg="#dddddd", activebackground="#3a3a3a", activeforeground="#ffffff",
        font=small_font, bd=0, highlightthickness=1, highlightbackground="#444444",
        relief="flat", padx=6, state="disabled",
    )
    model_menu["menu"].config(
        bg="#2a2a2a", fg="#dddddd", activebackground="#4a9a7a", activeforeground="#ffffff",
        font=small_font,
    )
    model_menu.pack(side="left")

    model_status_lbl = tk.Label(toolbar, textvariable=model_status_var,
                                fg="#e6a817", bg="#111111", font=badge_font)
    model_status_lbl.pack(side="left", padx=(8, 0))

    def _set_model_status(text: str, color: str) -> None:
        model_status_var.set(text)
        model_status_lbl.config(fg=color)

    model_holder: Optional[whisper_engine.ModelHolder] = None

    def _on_model_change(*_) -> None:
        if model_holder is None:
            return
        new_size = model_var.get()
        if new_size == model_holder.size:
            return
        model_menu.config(state="disabled")

        def _start(size: str) -> None:
            root.after(0, lambda: _set_model_status(f"⟳  Cargando {size}…", "#e6a817"))

        def _done(size: str) -> None:
            def _u() -> None:
                _set_model_status(f"✓  {size}", "#4a9a7a")
                model_menu.config(state="normal")
            root.after(0, _u)

        def _error(size: str, err: Exception) -> None:
            def _u() -> None:
                _set_model_status(f"✗  Error al cargar {size}", "#cc4444")
                model_var.set(model_holder.size)
                model_menu.config(state="normal")
            root.after(0, _u)

        model_holder.swap(new_size, on_start=_start, on_done=_done, on_error=_error)

    model_var.trace_add("write", _on_model_change)

    tk.Frame(toolbar, bg="#333333", width=1).pack(side="left", fill="y", padx=(12, 8))

    # Botón limpiar paneles
    def _clear_panels() -> None:
        for widget in (left_text, right_text):
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.config(state="disabled")
        nonlocal line_count
        line_count = 0
        _set_status("Paneles limpiados.", "#555555")

    tk.Button(
        toolbar, text="⌫ Limpiar", command=_clear_panels,
        bg="#2a2a2a", fg="#aaaaaa", activebackground="#3a3a3a", activeforeground="#ffffff",
        font=small_font, bd=0, highlightthickness=1, highlightbackground="#444444",
        relief="flat", padx=8, pady=2, cursor="hand2",
    ).pack(side="left", padx=(0, 8))

    tk.Frame(toolbar, bg="#333333", width=1).pack(side="left", fill="y", padx=(0, 8))

    # Toggle normalización
    normalize_evt = threading.Event()
    normalize_var = tk.BooleanVar(value=False)

    def _on_normalize_toggle() -> None:
        if normalize_var.get():
            normalize_evt.set()
        else:
            normalize_evt.clear()

    tk.Checkbutton(
        toolbar, text="Normalizar audio", variable=normalize_var,
        command=_on_normalize_toggle,
        fg="#888888", bg="#111111", selectcolor="#1a1a1a",
        activeforeground="#dddddd", activebackground="#111111",
        font=small_font, bd=0, highlightthickness=0, cursor="hand2",
    ).pack(side="left")

    norm_indicator = tk.Label(toolbar, text="○", fg="#555555", bg="#111111", font=badge_font)
    norm_indicator.pack(side="left", padx=(3, 0))

    def _update_norm_indicator(*_) -> None:
        norm_indicator.config(
            text="●" if normalize_var.get() else "○",
            fg="#e6a817" if normalize_var.get() else "#555555",
        )

    normalize_var.trace_add("write", _update_norm_indicator)

    tk.Frame(root, bg="#2a2a2a", height=1).pack(fill="x", side="top")

    # ── Paneles divididos ─────────────────────────────────────────────────────
    panels = tk.PanedWindow(root, orient="horizontal", bg="#333333",
                            sashwidth=3, sashrelief="flat", borderwidth=0)
    panels.pack(fill="both", expand=True)

    def _make_panel(parent, title: str, accent: str, text_color: str):
        frame = tk.Frame(parent, bg="#1a1a1a")
        tk.Label(frame, text=title, fg=accent, bg="#1a1a1a",
                 font=header_font, anchor="w", padx=8, pady=4).pack(fill="x")
        tk.Frame(frame, bg=accent, height=1).pack(fill="x", padx=8)
        tf = tk.Frame(frame, bg="#1a1a1a")
        tf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(tf, bg="#222222", troughcolor="#1a1a1a")
        sb.pack(side="right", fill="y")
        txt = tk.Text(
            tf, bg="#1a1a1a", fg=text_color, font=text_font, wrap="word",
            insertbackground="#1a1a1a", selectbackground="#333333",
            borderwidth=0, highlightthickness=0, padx=8, pady=6,
            yscrollcommand=sb.set, state="disabled", cursor="arrow", spacing3=4,
        )
        txt.tag_configure("timestamp", foreground="#555555", font=ts_font)
        txt.tag_configure("content",   foreground=text_color, font=text_font)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        return frame, txt

    # Panel izquierdo: TRANSCRIPCIÓN (lo que se escucha, cualquier idioma)
    left_frame,  left_text  = _make_panel(panels, "TRANSCRIPCIÓN", "#7799bb", "#cccccc")
    # Panel derecho: TRADUCCIÓN (siempre al idioma opuesto)
    right_frame, right_text = _make_panel(panels, "TRADUCCIÓN",    "#4a9a7a", "#7fdbca")

    panels.add(left_frame,  stretch="always")
    panels.add(right_frame, stretch="always")

    # ── Status bar ────────────────────────────────────────────────────────────
    status_var = tk.StringVar(value="Iniciando…")
    status_lbl = tk.Label(root, textvariable=status_var, fg="#e6a817", bg="#111111",
                          font=small_font, anchor="w", padx=8, pady=3)
    status_lbl.pack(fill="x", side="bottom")

    def _set_status(text: str, color: str = "#555555") -> None:
        status_var.set(text)
        status_lbl.config(fg=color)

    # ── Lógica de inserción en paneles ────────────────────────────────────────
    line_count = 0

    def _append(widget: tk.Text, text: str, ts: float) -> None:
        """Inserta una línea con timestamp y texto."""
        widget.config(state="normal")
        if widget.get("1.0", "end").strip():
            widget.insert("end", "\n")
        stamp = time.strftime("%H:%M:%S", time.localtime(ts))
        widget.insert("end", f"[{stamp}]  ", "timestamp")
        widget.insert("end", text, "content")
        widget.config(state="disabled")
        widget.see("end")

    def _trim(widget: tk.Text) -> None:
        if widget.get("1.0", "end").strip().count("\n") + 1 > max_lines:
            widget.config(state="normal")
            widget.delete("1.0", "2.0")
            widget.config(state="disabled")

    def pump() -> None:
        nonlocal line_count
        try:
            while True:
                item = q_sys.get_nowait()
                if item is None:
                    return
                if item.source and item.translated:
                    src_lang = item.source_lang or "en"
                    # Idioma destino = el opuesto al detectado
                    tgt_lang = "en" if src_lang == "es" else "es"

                    _append(left_text,  item.source,     item.ts)
                    _append(right_text, item.translated, item.ts)
                    line_count += 1
                    if line_count > max_lines:
                        _trim(left_text)
                        _trim(right_text)
                        line_count -= 1
                    ts_str = time.strftime("%H:%M:%S", time.localtime(item.ts))
                    _set_status(f"Última transcripción: {ts_str}")
        except queue.Empty:
            pass
        root.after(100, pump)

    def on_close() -> None:
        stop_evt.set()
        try:
            q_sys.put_nowait(None)
        except queue.Full:
            pass
        stream_sys.close()
        AudioStream.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.update_idletasks()
    panels.sash_place(0, root.winfo_width() // 2, 0)

    # ── Workers ───────────────────────────────────────────────────────────────

    def _start_deepgram_worker() -> None:
        stop_evt.clear()
        threading.Thread(
            target=worker_deepgram,
            args=(q_sys, stop_evt, deepgram_api_key, chunk_sec, stream_sys, normalize_evt),
            daemon=True,
        ).start()
        backend_badge_var.set("● DEEPGRAM")
        backend_badge.config(fg="#4a9a7a")
        whisper_label.pack_forget()
        model_menu.pack_forget()
        model_status_lbl.pack_forget()
        _set_status("Escuchando con Deepgram  (detecta EN y ES automáticamente)…", "#555555")
        pump()

    def _start_whisper_worker(mholder: whisper_engine.ModelHolder) -> None:
        stop_evt.clear()
        threading.Thread(
            target=worker_system,
            args=(q_sys, stop_evt, mholder, chunk_sec, vad_filter,
                  stream_sys, prompt, normalize_evt),
            daemon=True,
        ).start()
        backend_badge_var.set("● WHISPER")
        backend_badge.config(fg="#7799bb")
        _set_status("Escuchando con Whisper  (inglés)…", "#555555")
        pump()

    def _on_backend_change(*_) -> None:
        new_backend = backend_var.get()
        stop_evt.set()
        time.sleep(0.1)
        stop_evt.clear()

        if new_backend == "deepgram":
            if not deepgram_api_key:
                _set_status("ERROR: DEEPGRAM_API_KEY no configurado en .env", "#cc4444")
                backend_var.set("whisper")
                return
            whisper_label.pack_forget()
            model_menu.pack_forget()
            model_status_lbl.pack_forget()
            _start_deepgram_worker()
        else:
            whisper_label.pack(side="left", padx=(0, 4))
            model_menu.pack(side="left")
            model_status_lbl.pack(side="left", padx=(8, 0))
            if model_holder is not None:
                _start_whisper_worker(model_holder)
            else:
                _set_status("Cargando modelo Whisper…", "#e6a817")

    backend_var.trace_add("write", _on_backend_change)

    # ── Carga inicial ─────────────────────────────────────────────────────────
    def _load_initial() -> None:
        nonlocal model_holder
        if backend_var.get() == "deepgram":
            if not deepgram_api_key:
                root.after(0, lambda: _set_status(
                    "ERROR: DEEPGRAM_API_KEY no configurado en .env", "#cc4444"))
                return
            root.after(0, _start_deepgram_worker)
            return

        root.after(0, lambda: _set_status(f"Cargando modelo {effective_model}…", "#e6a817"))
        try:
            loaded = whisper_engine.load(effective_model)

            def _on_ready() -> None:
                nonlocal model_holder
                model_holder = whisper_engine.ModelHolder(loaded, effective_model)
                _set_model_status(f"✓  {effective_model}", "#4a9a7a")
                model_menu.config(state="normal")
                _start_whisper_worker(model_holder)

            root.after(0, _on_ready)
        except Exception as exc:
            root.after(0, lambda: _set_model_status("✗  Error al cargar modelo", "#cc4444"))
            root.after(0, lambda: _set_status(f"Error: {exc}", "#cc4444"))

    threading.Thread(target=_load_initial, daemon=True).start()

    root.mainloop()
    stop_evt.set()
