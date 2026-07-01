# Patch: Known6 Mask/Profile Layout v1
#
# Warum:
# Dein aktuelles Bild ist besser als vorher, aber noch nicht sauber genug.
# Der Grund ist ziemlich wahrscheinlich, dass das Known6-Layout immer noch stark
# von edge.type / Edge-Richtung abhängig ist. Bei den neuen Rundkanten werden diese
# Kanten teilweise falsch klassifiziert. Dadurch stimmen zwar die 2x3-Positionen
# ungefähr, aber die Teile greifen nicht sauber genug ineinander.
#
# Dieser Patch macht für das bekannte 6er-Puzzle zwei Dinge:
# 1) Rotation wird über die echte Maskengeometrie bestimmt:
#    Aussenkanten sollen flach sein, Innenkanten sollen nicht flach sein.
# 2) Translation wird über Seitenprofile aus den Pixelmasken bestimmt,
#    nicht über die alten Edge-Objekte.
#
# Ausführen im Repo-Hauptordner:
#     python fix_known6_profile_layout_v1.py
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
backup = path.with_suffix(".py.bak_known6_profile_layout_v1")
backup.write_text(orig, encoding="utf-8")

text = orig

# ------------------------------------------------------------
# 1) Config stabilisieren
# ------------------------------------------------------------
new_grid = """KNOWN_6PIECE_GRID = [
    [6, 2, 1],
    [3, 5, 4],
]"""

text, n_grid = re.subn(
    r"KNOWN_6PIECE_GRID\s*=\s*\[\s*\[[^\]]+\]\s*,\s*\[[^\]]+\]\s*,?\s*\]",
    new_grid,
    text,
    count=1,
    flags=re.S,
)
if n_grid == 0:
    raise RuntimeError("Konnte KNOWN_6PIECE_GRID nicht finden.")

# Wir ersetzen _build_known_6piece_edge_matched_layout durch Profil-Matching,
# deshalb muss der Known6-Pfad diese Funktion auch verwenden.
if "KNOWN_6PIECE_USE_EDGE_TRANSLATION" in text:
    text = re.sub(
        r"KNOWN_6PIECE_USE_EDGE_TRANSLATION\s*=\s*(True|False)",
        "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True",
        text,
        count=1,
    )
else:
    text = text.replace(new_grid, new_grid + "\nKNOWN_6PIECE_USE_EDGE_TRANSLATION = True", 1)

# Keine festen Rotationswerte erzwingen, weil diese die Geometrie verschieben.
if "KNOWN_6PIECE_FORCE_ROTATION_STEPS" in text:
    text = re.sub(
        r"KNOWN_6PIECE_FORCE_ROTATION_STEPS\s*=\s*True",
        "KNOWN_6PIECE_FORCE_ROTATION_STEPS = False",
        text,
    )

# Profil-Clearance ergänzen.
if "KNOWN_6PIECE_PROFILE_CLEARANCE_PX" not in text:
    marker = "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True"
    if marker in text:
        text = text.replace(
            marker,
            marker + "\n# Abstand zwischen passenden Silhouettenprofilen. 0.0 = möglichst eng.\nKNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0",
            1,
        )
    else:
        text = text.replace(new_grid, new_grid + "\nKNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0", 1)

# ------------------------------------------------------------
# 2) Maskenbasierte Rotation ersetzen
# ------------------------------------------------------------
rotation_block = r