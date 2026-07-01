# Patch: Known6 feste Rotationen wieder deaktivieren
#
# Warum:
# Der letzte Patch fix_known6_fixed_rotations_v2 greift direkt in
# _choose_known_grid_rotation(...) ein. Dadurch werden die Puzzleteile bereits
# im Solver-Layout anders gedreht. Weil das Layout die gedrehten Geometrien
# benutzt, ändern sich dadurch nicht nur die Winkel, sondern auch die Positionen.
#
# Ziel:
# - Known6-Raster bleibt aktiv
# - gute Axis-Snap/Compact-Positionierung bleibt aktiv
# - feste Known6-Rotationen werden deaktiviert
# - Roboter-/A5-Drehung soll wieder über shape_match + A5-Snap laufen
#
# Ausführen im Repo-Hauptordner:
#     python fix_known6_disable_fixed_rotations_v1.py
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
backup = path.with_suffix(".py.bak_disable_fixed_rotations_v1")
backup.write_text(orig, encoding="utf-8")

text = orig

# 1) Feste Rotation deaktivieren.
if "KNOWN_6PIECE_FORCE_ROTATION_STEPS" in text:
    text = re.sub(
        r"KNOWN_6PIECE_FORCE_ROTATION_STEPS\s*=\s*True",
        "KNOWN_6PIECE_FORCE_ROTATION_STEPS = False",
        text,
    )
else:
    # Falls die Config fehlt, explizit ergänzen.
    marker = "KNOWN_6PIECE_USE_EDGE_TRANSLATION"
    idx = text.find(marker)
    if idx != -1:
        line_end = text.find("\n", idx)
        text = text[:line_end + 1] + "KNOWN_6PIECE_FORCE_ROTATION_STEPS = False\n" + text[line_end + 1:]

# 2) Known6-Raster auf die zuletzt richtige ID-Anordnung setzen.
grid = """KNOWN_6PIECE_GRID = [
    [6, 2, 1],
    [3, 5, 4],
]"""
text, n_grid = re.subn(
    r"KNOWN_6PIECE_GRID\s*=\s*\[\s*\[[^\]]+\]\s*,\s*\[[^\]]+\]\s*,?\s*\]",
    grid,
    text,
    count=1,
    flags=re.S,
)
if n_grid == 0:
    raise RuntimeError("Konnte KNOWN_6PIECE_GRID nicht finden.")

# 3) Edge-Translation deaktiviert lassen, sonst kollabiert/verschiebt das Layout zu stark.
if "KNOWN_6PIECE_USE_EDGE_TRANSLATION" in text:
    text = re.sub(
        r"KNOWN_6PIECE_USE_EDGE_TRANSLATION\s*=\s*True",
        "KNOWN_6PIECE_USE_EDGE_TRANSLATION = False",
        text,
    )
else:
    text = text.replace(grid, grid + "\nKNOWN_6PIECE_USE_EDGE_TRANSLATION = False", 1)

# 4) SmallSolver-exportierte Rotation nicht für A5 benutzen.
# Das hatte vorher zu falschen C-Achsen-Werten geführt.
if "USE_SMALL_SOLVER_EXPORTED_ROTATION_FOR_A5" in text:
    text = re.sub(
        r"USE_SMALL_SOLVER_EXPORTED_ROTATION_FOR_A5\s*=\s*True",
        "USE_SMALL_SOLVER_EXPORTED_ROTATION_FOR_A5 = False",
        text,
    )

# 5) Optional: falls der Fixed-Rotation-Block in _choose_known_grid_rotation steckt,
# bleibt er wegen FORCE=False wirkungslos. Zur Kontrolle eine klare Ausgabe ergänzen.
if "[KNOWN6 FIXED ROT DISABLED]" not in text:
    marker = 'print(f"[KNOWN6 GRID] expected/current grid = {KNOWN_6PIECE_GRID}")'
    if marker in text:
        text = text.replace(
            marker,
            marker + '\n    print("[KNOWN6 FIXED ROT DISABLED] rotations come from _choose_known_grid_rotation")',
            1,
        )

path.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(path), doraise=True)
except Exception as exc:
    path.write_text(orig, encoding="utf-8")
    raise RuntimeError(f"Patch erzeugte Syntaxfehler. Datei wurde zurückgesetzt: {exc}")

print("[OK] Fixed Known6 rotations deaktiviert.")
print(f"[BACKUP] {backup}")
print("")
print("Jetzt testen:")
print("    python tests/run_robot_solver_small.py")
print("")
print("Erwartete Konsolenausgabe:")
print("    [KNOWN6 FIXED ROT DISABLED] rotations come from _choose_known_grid_rotation")
print("    [KNOWN6 MODE] axis-snap compact 2x3 layout")
print("")
print("Wichtig:")
print("- Wenn die Positionen danach wieder gut sind, NICHT mehr _choose_known_grid_rotation hardcoden.")
print("- Falls nur die Roboter-C-Achse falsch ist, muss man danach nur die finalen C-Rotation-Offsets anpassen, nicht das Solver-Layout.")
