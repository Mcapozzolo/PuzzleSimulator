"""
Erzeugt tests/run_robot_solver_small.py aus deinem bestehenden
run_robot_solver_tolerant.py oder run_robot_solver.py und schaltet den SmallSolver ein.

Benutzung im Repo-Hauptordner:
    python install_smallsolver_runner.py

Danach:
    python tests/run_robot_solver_small.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
TESTS = ROOT / "tests"

SOURCE_CANDIDATES = [
    TESTS / "run_robot_solver_tolerant.py",
    TESTS / "run_robot_solver.py",
    TESTS / "run_robot_solver_camera_optimized.py",
]

source_path = next((p for p in SOURCE_CANDIDATES if p.exists()), None)
if source_path is None:
    raise FileNotFoundError(
        "Keine Runner-Datei gefunden. Erwartet eine von: "
        + ", ".join(str(p) for p in SOURCE_CANDIDATES)
    )

target_path = TESTS / "run_robot_solver_small.py"
text = source_path.read_text(encoding="utf-8")

# 1) Projektroot robust setzen, falls noch nicht vorhanden.
project_root_block = '''\n# Damit Imports wie "from solver..." auch funktionieren, wenn das Skript aus tests/ gestartet wird.\nPROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\nif PROJECT_ROOT not in sys.path:\n    sys.path.insert(0, PROJECT_ROOT)\n'''
if "PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))" not in text:
    # nach den ersten import-Zeilen einsetzen, wenn os/sys importiert wurden
    marker = "import numpy as np\n"
    if marker in text:
        text = text.replace(marker, marker + project_root_block, 1)
    else:
        text = "import os\nimport sys\n" + project_root_block + "\n" + text

# 2) Config-Flag einfügen.
if "USE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE" not in text:
    marker = "USE_SCREW_HOLE_PICK = True"
    if marker in text:
        text = text.replace(
            marker,
            marker + "\nUSE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE = True  # neuer 6-Teile SmallSolver",
            1,
        )
    else:
        text = text.replace(
            "# =========================\n# CONFIG\n# =========================",
            "# =========================\n# CONFIG\n# =========================\nUSE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE = True  # neuer 6-Teile SmallSolver",
            1,
        )

# 3) solve_puzzle()-Aufruf ersetzen. Achtung: exakt nur den Solver-Aufruf ersetzen,
# nicht Funktionsnamen oder andere Vorkommen.
old_calls = [
    "puzzle.solve_puzzle()",
    "puzzle.solve_puzzle(fallback=True)",
]
new_call = '''\n    if USE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE and hasattr(puzzle, "solve_puzzle_small"):\n        print("[RUN] Löse Puzzle mit SmallSolver...")\n        puzzle.solve_puzzle_small(fallback=True)\n    else:\n        print("[RUN] Löse Puzzle mit normalem Solver...")\n        puzzle.solve_puzzle()'''

replaced = False
for old in old_calls:
    if old in text:
        text = text.replace(old, new_call, 1)
        replaced = True
        break
if not replaced:
    raise RuntimeError(
        "Konnte den Aufruf puzzle.solve_puzzle() nicht automatisch finden. "
        "Bitte manuell ersetzen."
    )

# 4) Sicherstellen, dass Solver-Debugbilder gespeichert werden.
# Falls deine Runner-Datei bereits genau diese Schleife hat, wird nichts geändert.
debug_save_block = '''\n    # SmallSolver / Puzzle-Debugbilder speichern\n    try:\n        for i, img in enumerate(puzzle.get_debug_images()):\n            out = os.path.join(DEBUG_DIR, f"{i + 10:02d}_debug.png")\n            cv2.imwrite(out, img)\n            print(f"[DEBUG] Saved {out}")\n    except Exception as exc:\n        print(f"[WARN] Solver-Debugbilder konnten nicht gespeichert werden: {exc}")\n'''

if "puzzle.get_debug_images()" not in text:
    # Direkt nach dem Solver-Block einfügen.
    text = text.replace(new_call, new_call + debug_save_block, 1)
else:
    # Bestehender Debug-Export existiert bereits. Nichts tun.
    pass

target_path.write_text(text, encoding="utf-8")
print(f"Erstellt: {target_path}")
print(f"Basisdatei: {source_path}")
print("Jetzt testen mit:")
print("    python tests/run_robot_solver_small.py")
