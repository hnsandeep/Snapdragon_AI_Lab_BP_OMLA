"""ui/app.py — Tkinter offline desktop UI for the Snapdragon AI Lab demo.

Features
────────
  • "OFFLINE MODE: ON  No network calls" badge — prominent, always visible
  • Live transcript scroll area — updates every 3 s
  • Language toggle (source + target dropdowns)
  • "Generate Summary" button — streams GenieX tokens into summary pane
  • Start / Stop recording buttons

Run:  python ui/app.py
"""
from __future__ import annotations
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, font as tkfont
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from pipeline import Pipeline, PipelineConfig, Segment

# ── Language catalogue ────────────────────────────────────────────────────────
LANGUAGES = [
    ("English",   "en", "eng_Latn"),
    ("Hindi",     "hi", "hin_Deva"),
    ("Kannada",   "kn", "kan_Knda"),
    ("Tamil",     "ta", "tam_Taml"),
    ("Telugu",    "te", "tel_Telu"),
    ("Bengali",   "bn", "ben_Beng"),
    ("Marathi",   "mr", "mar_Deva"),
    ("Gujarati",  "gu", "guj_Gujr"),
    ("Malayalam", "ml", "mal_Mlym"),
    ("German",    "de", "deu_Latn"),
    ("French",    "fr", "fra_Latn"),
    ("Spanish",   "es", "spa_Latn"),
]
LANG_DISPLAY  = [l[0] for l in LANGUAGES]
LANG_ISO1     = {l[0]: l[1] for l in LANGUAGES}
LANG_TAG      = {l[0]: l[2] for l in LANGUAGES}

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#1a1a2e"
PANEL       = "#16213e"
ACCENT      = "#0f3460"
GREEN       = "#00ff88"
RED_OFF     = "#ff4444"
TEXT_FG     = "#e0e0e0"
BADGE_BG    = "#003322"
BADGE_FG    = "#00ff88"
BTN_START   = "#006633"
BTN_STOP    = "#660000"
BTN_SUM     = "#2244aa"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Offline Multilingual Lecture Assistant — Snapdragon AI Lab")
        self.configure(bg=BG)
        self.geometry("1060x740")
        self.resizable(True, True)

        self._pipeline: Pipeline | None = None
        self._recording = False
        self._poll_id   = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        bold14 = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        bold11 = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        reg10  = tkfont.Font(family="Segoe UI", size=10)
        mono10 = tkfont.Font(family="Consolas",  size=10)

        # ── OFFLINE badge ──────────────────────────────────────────────────────
        badge_frame = tk.Frame(self, bg=BADGE_BG, bd=2, relief="ridge")
        badge_frame.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(
            badge_frame,
            text="🔒  OFFLINE MODE: ON  |  No network calls  |  All inference on-device (Snapdragon NPU)",
            bg=BADGE_BG, fg=BADGE_FG,
            font=bold11,
            pady=6,
        ).pack()

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="Offline Multilingual Lecture Assistant",
                 bg=BG, fg=TEXT_FG, font=bold14).pack(pady=(4, 0))

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill="x", padx=10, pady=6)

        tk.Label(ctrl, text="Source:", bg=BG, fg=TEXT_FG, font=reg10).pack(side="left", padx=(0, 4))
        self._src_var = tk.StringVar(value="English")
        ttk.Combobox(ctrl, textvariable=self._src_var, values=LANG_DISPLAY,
                     width=12, state="readonly").pack(side="left", padx=(0, 12))

        tk.Label(ctrl, text="→  Target:", bg=BG, fg=TEXT_FG, font=reg10).pack(side="left", padx=(0, 4))
        self._tgt_var = tk.StringVar(value="Hindi")
        ttk.Combobox(ctrl, textvariable=self._tgt_var, values=LANG_DISPLAY,
                     width=12, state="readonly").pack(side="left", padx=(0, 20))

        self._btn_start = tk.Button(ctrl, text="▶  Start Recording",
                                    bg=BTN_START, fg="white", font=bold11,
                                    relief="flat", padx=12, pady=4,
                                    command=self._start_recording)
        self._btn_start.pack(side="left", padx=4)

        self._btn_stop = tk.Button(ctrl, text="⏹  Stop",
                                   bg=BTN_STOP, fg="white", font=bold11,
                                   relief="flat", padx=12, pady=4,
                                   state="disabled", command=self._stop_recording)
        self._btn_stop.pack(side="left", padx=4)

        self._btn_sum = tk.Button(ctrl, text="✦  Generate Summary",
                                  bg=BTN_SUM, fg="white", font=bold11,
                                  relief="flat", padx=12, pady=4,
                                  command=self._generate_summary)
        self._btn_sum.pack(side="left", padx=4)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(ctrl, textvariable=self._status_var,
                 bg=BG, fg=GREEN, font=reg10).pack(side="left", padx=12)

        # ── Main split ────────────────────────────────────────────────────────
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=6, sashrelief="groove")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: transcript
        left = tk.Frame(paned, bg=PANEL)
        paned.add(left, minsize=400, stretch="always")

        tk.Label(left, text="Live Transcript", bg=PANEL, fg=TEXT_FG,
                 font=bold11).pack(anchor="w", padx=6, pady=(6, 2))
        self._transcript_box = scrolledtext.ScrolledText(
            left, bg="#0d0d1a", fg=TEXT_FG, font=mono10,
            wrap="word", state="disabled", relief="flat")
        self._transcript_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Right: summary
        right = tk.Frame(paned, bg=PANEL)
        paned.add(right, minsize=360, stretch="always")

        tk.Label(right, text="Summary (GenieX NPU)", bg=PANEL, fg=TEXT_FG,
                 font=bold11).pack(anchor="w", padx=6, pady=(6, 2))
        self._summary_box = scrolledtext.ScrolledText(
            right, bg="#0d0d1a", fg=TEXT_FG, font=mono10,
            wrap="word", state="disabled", relief="flat")
        self._summary_box.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # ── Bottom status bar ─────────────────────────────────────────────────
        bar = tk.Frame(self, bg=ACCENT)
        bar.pack(fill="x", side="bottom")
        self._seg_count_var = tk.StringVar(value="Segments: 0")
        tk.Label(bar, textvariable=self._seg_count_var,
                 bg=ACCENT, fg=TEXT_FG, font=reg10, padx=8).pack(side="left")
        tk.Label(bar, text="Powered by Qualcomm Snapdragon X Elite  |  QNN/NPU",
                 bg=ACCENT, fg=TEXT_FG, font=reg10, padx=8).pack(side="right")

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self._recording:
            return
        src = self._src_var.get()
        tgt = self._tgt_var.get()
        cfg = PipelineConfig(
            src_lang=LANG_ISO1[src],
            tgt_lang=LANG_TAG[tgt],
            translate_enabled=(src != tgt),
        )
        self._pipeline = Pipeline(cfg)
        self._pipeline.on_segment = self._on_segment
        self._pipeline.start()
        self._recording = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._status_var.set("🔴  Recording …")
        self._poll()

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        if self._poll_id:
            self.after_cancel(self._poll_id)
        if self._pipeline:
            self._pipeline.stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_var.set("Stopped.  Click 'Generate Summary' to summarize.")

    def _on_segment(self, seg: Segment) -> None:
        # called from worker thread — schedule UI update on main thread
        self.after(0, self._append_segment, seg)

    def _append_segment(self, seg: Segment) -> None:
        display = seg.text_tgt if seg.text_tgt else seg.text_src
        line    = f"[{seg.start_s:.1f}s – {seg.end_s:.1f}s]\n{display}\n\n"
        self._transcript_box.config(state="normal")
        self._transcript_box.insert("end", line)
        self._transcript_box.see("end")
        self._transcript_box.config(state="disabled")
        if self._pipeline:
            self._seg_count_var.set(f"Segments: {len(self._pipeline.get_segments())}")

    def _poll(self) -> None:
        """Periodic refresh (fallback if on_segment missed anything)."""
        if self._recording:
            self._poll_id = self.after(3000, self._poll)

    # ── Summary ───────────────────────────────────────────────────────────────

    def _generate_summary(self) -> None:
        if self._pipeline is None:
            self._status_var.set("No recording — start recording first.")
            return
        self._btn_sum.config(state="disabled")
        self._status_var.set("⚙  Generating summary on NPU …")
        self._summary_box.config(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.config(state="disabled")
        threading.Thread(target=self._run_summary, daemon=True).start()

    def _run_summary(self) -> None:
        def on_tok(tok: str) -> None:
            self.after(0, self._append_summary_token, tok)

        try:
            result = self._pipeline.summarize(on_token=on_tok)
            self.after(0, self._status_var.set, "✅  Summary complete.")
        except Exception as exc:
            result = f"[Error: {exc}]"
            self.after(0, self._status_var.set, f"Error: {exc}")

        self.after(0, self._btn_sum.config, {"state": "normal"})

    def _append_summary_token(self, tok: str) -> None:
        self._summary_box.config(state="normal")
        self._summary_box.insert("end", tok)
        self._summary_box.see("end")
        self._summary_box.config(state="disabled")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop_recording()
        if self._pipeline:
            self._pipeline.close()
        self.destroy()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
