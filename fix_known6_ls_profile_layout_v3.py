# Patch: Known6 Least-Squares Profile Layout v3
#
# Ziel:
# Dein aktuelles Resultat ist schon nahe dran, aber noch etwas ungenau.
# In v2 wurden die Profil-Matching-Positionen am Ende wieder stark auf
# gemeinsame Reihen/Spalten gemittelt. Dadurch gehen genaue Kantenanpassungen
# teilweise verloren.
#
# Dieser Patch ersetzt nur _build_known_6piece_edge_matched_layout(...):
# - alle bekannten Nachbarschaften werden zuerst mit Profil-Matching gemessen
# - danach werden die Teilpositionen per Least-Squares gemeinsam optimiert
# - keine harte Zeilen-/Spalten-Mittelung mehr
#
# Ausführen im Repo-Hauptordner:
#     python fix_known6_ls_profile_layout_v3.py
#
# Danach:
#     python tests/run_robot_solver_small.py

from __future__ import annotations

from pathlib import Path
import re
import py_compile

path = Path("tests/run_robot_solver_small.py")
if not path.exists():
    raise FileNotFoundError("tests/run_robot_solver_small.py nicht gefunden. Bitte im Repo-Hauptordner ausführen.")

orig = path.read_text(encoding="utf-8")
backup = path.with_suffix(".py.bak_known6_ls_profile_layout_v3")
backup.write_text(orig, encoding="utf-8")

text = orig

# Sicherheit: Known6 aktiv und Profil-Layout aktiv lassen.
if "KNOWN_6PIECE_USE_EDGE_TRANSLATION" in text:
    text = re.sub(
        r"KNOWN_6PIECE_USE_EDGE_TRANSLATION\s*=\s*(True|False)",
        "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True",
        text,
        count=1,
    )

# 0.0 bleibt am besten für das Bild-Debug: zuerst exakt passend berechnen.
# Falls physisch zu eng: später auf 1.0 oder 2.0 setzen.
if "KNOWN_6PIECE_PROFILE_CLEARANCE_PX" in text:
    text = re.sub(
        r"KNOWN_6PIECE_PROFILE_CLEARANCE_PX\s*=\s*[-+]?\d+(\.\d+)?",
        "KNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0",
        text,
        count=1,
    )

new_func = r