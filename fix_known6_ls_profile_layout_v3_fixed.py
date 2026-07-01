# Patch: Known6 Least-Squares Profile Layout v3 FIXED
#
# Fix fuer deinen Fehler:
#     NameError: name 'r' is not defined
#
# Ursache:
# Die vorherige Patch-Datei wurde beim Erzeugen abgeschnitten, weil der
# eingebettete raw-string falsch verschachtelt war.
#
# Ausfuehren im Repo-Hauptordner:
#     python fix_known6_ls_profile_layout_v3_fixed.py
#
# Danach:
#     python tests/run_robot_solver_small.py

from __future__ import annotations

from pathlib import Path
import re
import py_compile

path = Path("tests/run_robot_solver_small.py")
if not path.exists():
    raise FileNotFoundError("tests/run_robot_solver_small.py nicht gefunden. Bitte im Repo-Hauptordner ausfuehren.")

orig = path.read_text(encoding="utf-8")
backup = path.with_suffix(".py.bak_known6_ls_profile_layout_v3_fixed")
backup.write_text(orig, encoding="utf-8")

text = orig

# Known6 aktiv und Profil-Layout aktiv lassen.
if "KNOWN_6PIECE_USE_EDGE_TRANSLATION" in text:
    text = re.sub(
        r"KNOWN_6PIECE_USE_EDGE_TRANSLATION\s*=\s*(True|False)",
        "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True",
        text,
        count=1,
    )

# Clearance zuerst exakt lassen. Falls physisch zu eng: spaeter 1.0 oder 2.0 testen.
if "KNOWN_6PIECE_PROFILE_CLEARANCE_PX" in text:
    text = re.sub(
        r"KNOWN_6PIECE_PROFILE_CLEARANCE_PX\s*=\s*[-+]?\d+(\.\d+)?",
        "KNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0",
        text,
        count=1,
    )

new_func = '''
def _build_known_6piece_edge_matched_layout(pieces_by_id, rotations):
    # Known6 v3:
    # Profil-Matching wird nicht mehr nur sequentiell angewendet und danach
    # hart auf Reihen/Spalten gemittelt. Stattdessen werden alle gemessenen
    # Nachbarschafts-Vektoren gemeinsam per Least-Squares geloest.
    #
    # Vorteil:
    # - kleine Fehler einzelner Kanten werden verteilt
    # - Bottom-Row und Top-Row beeinflussen sich gegenseitig
    # - genaue Profilinformationen gehen nicht durch hartes Grid-Averaging verloren

    grid = KNOWN_6PIECE_GRID
    rows = len(grid)
    cols = len(grid[0])

    geoms = {
        int(pid): _make_known_grid_oriented_geometry(pieces_by_id[int(pid)], rotations[int(pid)])
        for row in grid
        for pid in row
    }

    # Bekannte 2x3-Nachbarschaften.
    adj = []
    for r in range(rows):
        for c in range(cols - 1):
            adj.append((int(grid[r][c]), int(grid[r][c + 1]), Directions.E, Directions.W))
    for r in range(rows - 1):
        for c in range(cols):
            adj.append((int(grid[r][c]), int(grid[r + 1][c]), Directions.S, Directions.N))

    # 1) Lokale Soll-Translationen messen:
    #    translation[B] - translation[A] = delta_ab
    constraints = []
    for a_pid, b_pid, a_dir, b_dir in adj:
        delta_ab, score = _best_translation_between_side_profiles(
            geoms[a_pid],
            geoms[b_pid],
            a_dir,
            b_dir,
        )

        if not np.all(np.isfinite(delta_ab)) or not np.isfinite(score):
            print(
                f"[KNOWN6 LS WARN] seam {a_pid}.{a_dir.name}->{b_pid}.{b_dir.name} "
                f"ungueltig, wird ignoriert"
            )
            continue

        # Gewicht: gute Profilmatches staerker, schlechte schwaecher.
        weight = 1.0 / max(1.0, float(score))
        constraints.append((a_pid, b_pid, np.asarray(delta_ab, dtype=np.float32), weight))

        print(
            f"[KNOWN6 LS MEASURE] {a_pid}.{a_dir.name} -> {b_pid}.{b_dir.name}: "
            f"delta=({delta_ab[0]:.1f},{delta_ab[1]:.1f}) score={score:.2f} weight={weight:.3f}"
        )

    if len(constraints) < 3:
        raise RuntimeError(f"KNOWN6 LS Layout hat zu wenige gueltige Constraints: {len(constraints)}")

    # 2) Least-Squares loesen.
    #    Pro Achse separat:
    #       T_b - T_a = delta
    #    Anchor: erstes Teil bleibt bei (0,0), damit das System eindeutig ist.
    pids = sorted(geoms.keys())
    idx_by_pid = {pid: i for i, pid in enumerate(pids)}
    n = len(pids)

    A_rows = []
    b_row = []
    b_col = []

    for a_pid, b_pid, delta_ab, weight in constraints:
        w = float(np.sqrt(weight))
        line = np.zeros(n, dtype=np.float64)
        line[idx_by_pid[b_pid]] = w
        line[idx_by_pid[a_pid]] = -w

        A_rows.append(line)
        b_row.append(float(delta_ab[0]) * w)
        b_col.append(float(delta_ab[1]) * w)

    # Anchor-Constraint: oberes linkes Teil auf 0,0 halten.
    anchor_pid = int(grid[0][0])
    anchor_weight = 10.0
    line = np.zeros(n, dtype=np.float64)
    line[idx_by_pid[anchor_pid]] = anchor_weight
    A_rows.append(line)
    b_row.append(0.0)
    b_col.append(0.0)

    A = np.vstack(A_rows)
    br = np.asarray(b_row, dtype=np.float64)
    bc = np.asarray(b_col, dtype=np.float64)

    sol_r, *_ = np.linalg.lstsq(A, br, rcond=None)
    sol_c, *_ = np.linalg.lstsq(A, bc, rcond=None)

    translations = {
        pid: np.asarray([float(sol_r[idx_by_pid[pid]]), float(sol_c[idx_by_pid[pid]])], dtype=np.float32)
        for pid in pids
    }

    # 3) Plausibilitaetscheck: untere Reihe muss unter oberer Reihe bleiben.
    top_ids = [int(x) for x in grid[0]]
    bottom_ids = [int(x) for x in grid[1]]
    top_mean_r = float(np.mean([translations[pid][0] for pid in top_ids]))
    bottom_mean_r = float(np.mean([translations[pid][0] for pid in bottom_ids]))

    if bottom_mean_r <= top_mean_r:
        desired_gap = 0.70 * np.mean([float(geoms[pid]["height"]) for pid in top_ids])
        correction = (top_mean_r + desired_gap) - bottom_mean_r
        for pid in bottom_ids:
            translations[pid][0] += float(correction)
        print(f"[KNOWN6 LS FIX] bottom row nach unten korrigiert um {correction:.1f}px")

    # 4) In den positiven Debugbereich verschieben.
    all_rows = []
    all_cols = []
    for pid, geom in geoms.items():
        t = translations[pid]
        for row_f, col_f, _ in geom["pixels_float"]:
            all_rows.append(t[0] + row_f)
            all_cols.append(t[1] + col_f)

    shift = np.asarray([
        float(KNOWN_6PIECE_GRID_MARGIN_PX) - min(all_rows),
        float(KNOWN_6PIECE_GRID_MARGIN_PX) - min(all_cols),
    ], dtype=np.float32)

    for pid in translations:
        translations[pid] = translations[pid] + shift

    # 5) Debug: Restfehler der Constraints ausgeben.
    for a_pid, b_pid, delta_ab, weight in constraints:
        actual = translations[b_pid] - translations[a_pid]
        err = actual - delta_ab
        print(
            f"[KNOWN6 LS ERR] {a_pid}->{b_pid}: "
            f"actual=({actual[0]:.1f},{actual[1]:.1f}) "
            f"target=({delta_ab[0]:.1f},{delta_ab[1]:.1f}) "
            f"err=({err[0]:+.1f},{err[1]:+.1f})"
        )

    for pid in sorted(translations):
        t = translations[pid]
        print(f"[KNOWN6 LS FINAL] P{pid} translation=({t[0]:.1f},{t[1]:.1f})")

    return geoms, translations
'''

start = text.find("def _build_known_6piece_edge_matched_layout(")
if start == -1:
    raise RuntimeError("Konnte _build_known_6piece_edge_matched_layout nicht finden.")

candidates = []
for marker in [
    "\ndef _known6_sync_pixels_to_edge_bbox",
    "\ndef solve_known_6piece_grid_if_applicable",
    "\ndef _known6_build_compact_constrained_layout",
]:
    idx = text.find(marker, start + 1)
    if idx != -1:
        candidates.append(idx)

if not candidates:
    raise RuntimeError("Konnte Ende von _build_known_6piece_edge_matched_layout nicht finden.")

end = min(candidates)
text = text[:start] + new_func.strip() + "\n\n" + text[end:]

# Debugmeldung anpassen.
text = text.replace(
    'print("[KNOWN6 MODE] mask/profile matched layout")',
    'print("[KNOWN6 MODE] least-squares profile layout v3 fixed")',
)
text = text.replace(
    'print("[KNOWN6 MODE] least-squares profile layout v3")',
    'print("[KNOWN6 MODE] least-squares profile layout v3 fixed")',
)

if "[KNOWN6 MODE] least-squares profile layout v3 fixed" not in text:
    marker = 'print(f"[KNOWN6 GRID] expected/current grid = {KNOWN_6PIECE_GRID}")'
    if marker in text:
        text = text.replace(
            marker,
            marker + '\n    print("[KNOWN6 MODE] least-squares profile layout v3 fixed")',
            1,
        )

path.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(path), doraise=True)
except Exception as exc:
    path.write_text(orig, encoding="utf-8")
    raise RuntimeError(f"Patch erzeugte Syntaxfehler. Datei wurde zurueckgesetzt: {exc}")

print("[OK] Known6 Least-Squares Profile Layout v3 fixed installiert.")
print(f"[BACKUP] {backup}")
print("")
print("Jetzt testen:")
print("    python tests/run_robot_solver_small.py")
print("")
print("Erwartete Konsolenausgabe:")
print("    [KNOWN6 MODE] least-squares profile layout v3 fixed")
print("    [KNOWN6 LS MEASURE] ...")
print("    [KNOWN6 LS ERR] ...")
