"""
Wettbewerb-Start-GUI fuer Raspberry Pi Touchscreen.
Zeigt einen grossen "Start"-Knopf. Beim Druecken wird der Solver gestartet
und die Terminal-Ausgabe live im Fenster angezeigt.
Ausfuehren:  python start_gui.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

SOLVER_CMD = [sys.executable, "tests/run_robot_solver_small.py"]
SCRIPT_DIR = Path(__file__).parent


def start_solver(button: tk.Button, log_text: tk.Text) -> None:
    button.config(state="disabled", text="Laeuft...")

    # Clear previous output
    log_text.config(state="normal")
    log_text.delete("1.0", tk.END)
    log_text.config(state="disabled")

    def run():
        try:
            proc = subprocess.Popen(
                SOLVER_CMD,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                _append_line(log_text, line)
            proc.wait()
            code = proc.returncode
            msg = "\n[Fertig!]\n" if code == 0 else f"\n[Fehler – Code {code}]\n"
        except Exception as exc:
            msg = f"\n[Fehler: {exc}]\n"

        _append_line(log_text, msg)
        button.config(state="normal", text="Start")

    threading.Thread(target=run, daemon=True).start()


def _append_line(log_text: tk.Text, line: str) -> None:
    """Thread-safe: append text and auto-scroll to bottom."""
    def _do():
        log_text.config(state="normal")
        log_text.insert(tk.END, line)
        log_text.see(tk.END)
        log_text.config(state="disabled")
    log_text.after(0, _do)


def main() -> None:
    root = tk.Tk()
    root.title("PREN – Puzzle Solver")
    root.attributes("-fullscreen", True)
    root.configure(bg="#1a1a2e")
    root.bind("<Escape>", lambda e: root.destroy())

    # ── Top bar ──────────────────────────────────────────────────────────────
    top = tk.Frame(root, bg="#1a1a2e")
    top.pack(fill="x", padx=20, pady=(20, 5))

    tk.Label(
        top,
        text="PREN Puzzle Solver",
        font=("Helvetica", 24, "bold"),
        bg="#1a1a2e",
        fg="#e0e0e0",
    ).pack(side="left")

    btn = tk.Button(
        top,
        text="Start",
        font=("Helvetica", 36, "bold"),
        bg="#16a34a",
        fg="white",
        activebackground="#15803d",
        activeforeground="white",
        relief="flat",
        padx=30,
        pady=10,
        cursor="hand2",
    )
    btn.pack(side="right")

    tk.Button(
        top,
        text="Beenden",
        font=("Helvetica", 14),
        bg="#7f1d1d",
        fg="white",
        activebackground="#991b1b",
        relief="flat",
        padx=12,
        pady=8,
        command=root.destroy,
    ).pack(side="right", padx=(0, 15))

    # ── Log output area ───────────────────────────────────────────────────────
    frame = tk.Frame(root, bg="#0f0f23")
    frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    log_text = tk.Text(
        frame,
        font=("Courier", 11),
        bg="#0f0f23",
        fg="#d4d4d4",
        insertbackground="white",
        yscrollcommand=scrollbar.set,
        wrap="word",
        state="disabled",
        relief="flat",
        padx=8,
        pady=6,
    )
    log_text.pack(fill="both", expand=True)
    scrollbar.config(command=log_text.yview)

    btn.config(command=lambda: start_solver(btn, log_text))

    root.mainloop()


if __name__ == "__main__":
    main()