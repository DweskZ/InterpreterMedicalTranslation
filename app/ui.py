"""Tkinter overlay UI with split panels and tabbed interface."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from app.audio import AudioStream, pick_loopback
from app.workers import CaptionLine, worker_system
from app import whisper_engine


def run(
    model_size: str,
    chunk_sec: float,
    max_lines: int,
    vad_filter: bool,
    device_hint: Optional[str],
    mic_hint: Optional[str],
    prompt: str,
) -> None:
    import tkinter as tk
    from tkinter import font as tkfont

    q_sys: queue.Queue[Optional[CaptionLine]] = queue.Queue(maxsize=64)
    stop_evt = threading.Event()

    # Modelo Whisper (.en -> multilenguaje cuando se habilite el mic tab)
    effective_model = model_size
    # Cuando el tab del mic se habilite, descomentar esto:
    # if model_size.endswith(".en"):
    #     effective_model = model_size.replace(".en", "")
    #     print(f"Usando modelo multilenguaje '{effective_model}' (necesario para ES+EN).", flush=True)
    model = whisper_engine.load(effective_model)

    # Audio stream (loopback del sistema)
    loopback_dev = pick_loopback(device_hint)
    stream_sys = AudioStream(loopback_dev)
    print(f"Dispositivo: {loopback_dev['name']}", flush=True)

    # --- Window ---
    root = tk.Tk()
    root.title("Clinic Translate - Medical Interpreter")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.93)
    root.configure(bg="#1a1a1a")
    root.geometry("1100x500+40+40")
    root.minsize(700, 300)

    header_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
    text_font = tkfont.Font(family="Segoe UI", size=13)
    ts_font = tkfont.Font(family="Consolas", size=10)
    small_font = tkfont.Font(family="Segoe UI", size=10)

    # --- Split panels ---
    panels = tk.PanedWindow(root, orient="horizontal", bg="#333333",
                            sashwidth=3, sashrelief="flat", borderwidth=0)
    panels.pack(fill="both", expand=True)

    def _make_panel(parent, title: str, title_color: str, text_color: str):
        frame = tk.Frame(parent, bg="#1a1a1a")
        tk.Label(frame, text=title, fg=title_color, bg="#1a1a1a",
                 font=header_font, anchor="w", padx=8, pady=4).pack(fill="x")
        tk.Frame(frame, bg=title_color, height=1).pack(fill="x", padx=8)

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
        txt.tag_configure("content", foreground=text_color, font=text_font)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        return frame, txt

    en_frame, en_text = _make_panel(panels, "ENGLISH (patient)", "#7799bb", "#cccccc")
    es_frame, es_text = _make_panel(panels, "SPANISH (translation)", "#4a9a7a", "#7fdbca")
    panels.add(en_frame, stretch="always")
    panels.add(es_frame, stretch="always")

    # --- Status bar ---
    status_var = tk.StringVar(value="Cargando...")
    tk.Label(root, textvariable=status_var, fg="#555555", bg="#111111",
             font=small_font, anchor="w", padx=8, pady=3).pack(fill="x", side="bottom")

    # --- Append logic ---
    line_count = 0

    def _append(widget: tk.Text, text: str, ts: float) -> None:
        widget.config(state="normal")
        if widget.get("1.0", "end").strip():
            widget.insert("end", "\n")
        stamp = time.strftime("%H:%M:%S", time.localtime(ts))
        widget.insert("end", f"[{stamp}] ", "timestamp")
        widget.insert("end", text, "content")
        widget.config(state="disabled")
        widget.see("end")

    def _trim(widget: tk.Text) -> None:
        content = widget.get("1.0", "end")
        if content.strip().count("\n") + 1 > max_lines:
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
                    _append(en_text, item.source, item.ts)
                    _append(es_text, item.translated, item.ts)
                    line_count += 1
                    if line_count > max_lines:
                        _trim(en_text)
                        _trim(es_text)
                        line_count -= 1
                    ts = time.strftime("%H:%M:%S", time.localtime(item.ts))
                    status_var.set(f"Ultima transcripcion: {ts}")
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
    status_var.set("Escuchando...")

    # Center sash
    root.update_idletasks()
    panels.sash_place(0, root.winfo_width() // 2, 0)

    # Worker
    th = threading.Thread(
        target=worker_system,
        args=(q_sys, stop_evt, model, chunk_sec, vad_filter, stream_sys, prompt),
        daemon=True,
    )
    th.start()

    pump()
    root.mainloop()
    stop_evt.set()
