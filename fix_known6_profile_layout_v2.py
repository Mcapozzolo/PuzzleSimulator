# Patch: Known6 Profile Layout v2
#
# Fix fuer Fehler:
#     NameError: name 'r' is not defined
#
# Ursache:
# Die vorherige Patch-Datei wurde beim Erzeugen abgeschnitten und enthielt
# versehentlich nur:
#     rotation_block = r
#
# Ausfuehren im Repo-Hauptordner:
#     python fix_known6_profile_layout_v2.py
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
backup = path.with_suffix(".py.bak_known6_profile_layout_v2")
backup.write_text(orig, encoding="utf-8")

text = orig

# ---------------------------------------------------------------------
# 1) Known6-Config stabilisieren
# ---------------------------------------------------------------------
grid = (
    "KNOWN_6PIECE_GRID = [\n"
    "    [6, 2, 1],\n"
    "    [3, 5, 4],\n"
    "]"
)

text, n_grid = re.subn(
    r"KNOWN_6PIECE_GRID\s*=\s*\[\s*\[[^\]]+\]\s*,\s*\[[^\]]+\]\s*,?\s*\]",
    grid,
    text,
    count=1,
    flags=re.S,
)

if n_grid == 0:
    raise RuntimeError("Konnte KNOWN_6PIECE_GRID nicht finden.")

if "KNOWN_6PIECE_USE_EDGE_TRANSLATION" in text:
    text = re.sub(
        r"KNOWN_6PIECE_USE_EDGE_TRANSLATION\s*=\s*(True|False)",
        "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True",
        text,
        count=1,
    )
else:
    text = text.replace(grid, grid + "\nKNOWN_6PIECE_USE_EDGE_TRANSLATION = True", 1)

if "KNOWN_6PIECE_FORCE_ROTATION_STEPS" in text:
    text = re.sub(
        r"KNOWN_6PIECE_FORCE_ROTATION_STEPS\s*=\s*True",
        "KNOWN_6PIECE_FORCE_ROTATION_STEPS = False",
        text,
    )

if "KNOWN_6PIECE_PROFILE_CLEARANCE_PX" not in text:
    marker = "KNOWN_6PIECE_USE_EDGE_TRANSLATION = True"
    if marker in text:
        text = text.replace(
            marker,
            marker + "\nKNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0",
            1,
        )
    else:
        text = text.replace(grid, grid + "\nKNOWN_6PIECE_PROFILE_CLEARANCE_PX = 0.0", 1)

# ---------------------------------------------------------------------
# 2) Maskenbasierte Rotation
# ---------------------------------------------------------------------
rotation_block = '''
def _side_profile_from_geometry(geom, direction):
    # Seitenprofil direkt aus den Pixeln.
    # N/S: Profilachse = col, Wert = row_edge
    # W/E: Profilachse = row, Wert = col_edge
    pts = np.asarray([(r, c) for r, c, _ in geom["pixels_float"]], dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rows_i = np.round(pts[:, 0]).astype(np.int32)
    cols_i = np.round(pts[:, 1]).astype(np.int32)

    prof = []
    if direction in (Directions.W, Directions.E):
        for rr in range(int(rows_i.min()), int(rows_i.max()) + 1):
            cs = cols_i[rows_i == rr]
            if cs.size == 0:
                continue
            edge_c = int(cs.min()) if direction == Directions.W else int(cs.max())
            prof.append((float(rr), float(edge_c)))
    else:
        for cc in range(int(cols_i.min()), int(cols_i.max()) + 1):
            rs = rows_i[cols_i == cc]
            if rs.size == 0:
                continue
            edge_r = int(rs.min()) if direction == Directions.N else int(rs.max())
            prof.append((float(cc), float(edge_r)))

    return np.asarray(prof, dtype=np.float32)


def _profile_flatness_score(profile):
    # Kleiner Wert = Seite ist gerade / Randkante.
    p = np.asarray(profile, dtype=np.float32)
    if p.shape[0] < 10:
        return 1e6

    vals = p[:, 1]
    lo, hi = np.percentile(vals, [10, 90])
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size < 5:
        core = vals

    return float(np.std(core))


def _score_known_grid_rotation(piece, row_idx, col_idx, rows, cols, rotation_steps):
    # Rotation ueber die echte Silhouette statt ueber edge.type.
    geom = _make_known_grid_oriented_geometry(piece, rotation_steps)
    external = _known_grid_external_dirs(row_idx, col_idx, rows, cols)

    score = 0.0
    for direction in [Directions.N, Directions.E, Directions.S, Directions.W]:
        flat = _profile_flatness_score(_side_profile_from_geometry(geom, direction))

        if direction in external:
            score += flat
        else:
            score += max(0.0, 18.0 - flat) * 2.0

    score += rotation_steps * 0.01
    return score


def _choose_known_grid_rotation(piece, row_idx, col_idx, rows, cols):
    # Falls irgendwo noch feste Rotationen aktiv waeren, bewusst ignorieren:
    # Die festen Werte haben bei dir die Positionen wieder verschlechtert.
    scored = []
    for steps in range(4):
        scored.append((_score_known_grid_rotation(piece, row_idx, col_idx, rows, cols, steps), steps))

    scored.sort(key=lambda item: item[0])
    best_score, best_steps = scored[0]

    print(
        f"[KNOWN6 ROTMASK] Piece {getattr(piece, 'id', '?')} cell=({row_idx},{col_idx}) "
        f"rotation_steps={best_steps} rot={best_steps * 90}deg score={best_score:.2f} "
        f"all={[(round(s, 1), r) for s, r in scored]}"
    )
    return int(best_steps)

'''

start = text.find("def _score_known_grid_rotation(")
end = text.find("\ndef _rotated_piece_bbox_dims", start)

if start == -1 or end == -1:
    # Fallback: nur _choose_known_grid_rotation ersetzen, falls der Score-Block anders aussieht.
    start = text.find("def _choose_known_grid_rotation(")
    if start == -1:
        raise RuntimeError("Konnte _choose_known_grid_rotation nicht finden.")
    candidates = [
        text.find("\ndef _rotated_piece_bbox_dims", start),
        text.find("\ndef _build_known_6piece_edge_matched_layout", start),
        text.find("\ndef solve_known_6piece_grid_if_applicable", start),
    ]
    candidates = [x for x in candidates if x != -1]
    if not candidates:
        raise RuntimeError("Konnte Ende von _choose_known_grid_rotation nicht finden.")
    end = min(candidates)

# Wenn _side_profile_from_geometry unmittelbar vorher schon existiert, ebenfalls mit ersetzen.
side_start = text.find("def _side_profile_from_geometry(")
if side_start != -1 and side_start < start:
    start = side_start

text = text[:start] + rotation_block.strip() + "\n\n" + text[end:]

# ---------------------------------------------------------------------
# 3) Profil-basiertes Layout ersetzt die alte Edge-Translation
# ---------------------------------------------------------------------
layout_block = '''
def _interp_profile(profile, x):
    p = np.asarray(profile, dtype=np.float32)
    if p.shape[0] < 2:
        return None

    order = np.argsort(p[:, 0])
    p = p[order]

    if x < p[0, 0] or x > p[-1, 0]:
        return None

    return float(np.interp(x, p[:, 0], p[:, 1]))


def _best_translation_between_side_profiles(block_geom, new_geom, placed_dir, new_dir):
    # Translation ueber echte Seitenprofile aus der Pixelmaske.
    a = _side_profile_from_geometry(block_geom, placed_dir)
    b = _side_profile_from_geometry(new_geom, new_dir)

    if a.shape[0] < 10 or b.shape[0] < 10:
        return np.array([0.0, 0.0], dtype=np.float32), float("inf")

    axis_a_min, axis_a_max = float(a[:, 0].min()), float(a[:, 0].max())
    axis_b_min, axis_b_max = float(b[:, 0].min()), float(b[:, 0].max())
    len_a = axis_a_max - axis_a_min
    len_b = axis_b_max - axis_b_min

    scan_min = axis_a_min - axis_b_max - 25.0
    scan_max = axis_a_max - axis_b_min + 25.0

    best_score = float("inf")
    best_t = np.array([0.0, 0.0], dtype=np.float32)

    for along_shift in np.linspace(scan_min, scan_max, 141):
        lo = max(axis_a_min, axis_b_min + along_shift)
        hi = min(axis_a_max, axis_b_max + along_shift)
        overlap = hi - lo

        if overlap < 0.35 * min(len_a, len_b):
            continue

        xs = np.linspace(lo, hi, 80)
        offsets = []
        for x in xs:
            av = _interp_profile(a, x)
            bv = _interp_profile(b, x - along_shift)
            if av is None or bv is None:
                continue
            offsets.append(av - bv)

        if len(offsets) < 25:
            continue

        offsets = np.asarray(offsets, dtype=np.float32)
        clearance = float(globals().get("KNOWN_6PIECE_PROFILE_CLEARANCE_PX", 0.0))

        if placed_dir == Directions.E and new_dir == Directions.W:
            perp_shift = float(np.median(offsets + clearance))
            residual = float(np.mean(np.abs(offsets + clearance - perp_shift)))
            t = np.array([along_shift, perp_shift], dtype=np.float32)

        elif placed_dir == Directions.W and new_dir == Directions.E:
            perp_shift = float(np.median(offsets - clearance))
            residual = float(np.mean(np.abs(offsets - clearance - perp_shift)))
            t = np.array([along_shift, perp_shift], dtype=np.float32)

        elif placed_dir == Directions.S and new_dir == Directions.N:
            perp_shift = float(np.median(offsets + clearance))
            residual = float(np.mean(np.abs(offsets + clearance - perp_shift)))
            t = np.array([perp_shift, along_shift], dtype=np.float32)

        elif placed_dir == Directions.N and new_dir == Directions.S:
            perp_shift = float(np.median(offsets - clearance))
            residual = float(np.mean(np.abs(offsets - clearance - perp_shift)))
            t = np.array([perp_shift, along_shift], dtype=np.float32)

        else:
            continue

        score = residual + max(0.0, 0.65 * min(len_a, len_b) - overlap) * 0.05

        if score < best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def _build_known_6piece_edge_matched_layout(pieces_by_id, rotations):
    # Name bleibt gleich, Inhalt ist Profil-Matching.
    grid = KNOWN_6PIECE_GRID
    rows = len(grid)
    cols = len(grid[0])

    geoms = {
        int(pid): _make_known_grid_oriented_geometry(pieces_by_id[int(pid)], rotations[int(pid)])
        for row in grid
        for pid in row
    }

    translations = {int(grid[0][0]): np.array([0.0, 0.0], dtype=np.float32)}

    # Kontrollierte Reihenfolge:
    # oben links -> oben mitte -> oben rechts
    # dann untere Reihe an obere Reihe
    # dann untere Reihe horizontal.
    adj = []
    for c in range(cols - 1):
        adj.append((int(grid[0][c]), int(grid[0][c + 1]), Directions.E, Directions.W))
    for c in range(cols):
        adj.append((int(grid[0][c]), int(grid[1][c]), Directions.S, Directions.N))
    for c in range(cols - 1):
        adj.append((int(grid[1][c]), int(grid[1][c + 1]), Directions.E, Directions.W))

    changed = True
    while changed and len(translations) < len(geoms):
        changed = False

        for placed_pid, new_pid, placed_dir, new_dir in adj:
            if placed_pid in translations and new_pid not in translations:
                t_local, score = _best_translation_between_side_profiles(
                    geoms[placed_pid], geoms[new_pid], placed_dir, new_dir
                )
                t = translations[placed_pid] + t_local
                translations[new_pid] = t
                print(
                    f"[KNOWN6 PROFILE] {placed_pid}.{placed_dir.name} -> {new_pid}.{new_dir.name}: "
                    f"translation=({t[0]:.1f},{t[1]:.1f}) score={score:.2f}"
                )
                changed = True

            elif new_pid in translations and placed_pid not in translations:
                t_local, score = _best_translation_between_side_profiles(
                    geoms[new_pid], geoms[placed_pid], new_dir, placed_dir
                )
                t = translations[new_pid] + t_local
                translations[placed_pid] = t
                print(
                    f"[KNOWN6 PROFILE] {new_pid}.{new_dir.name} -> {placed_pid}.{placed_dir.name}: "
                    f"translation=({t[0]:.1f},{t[1]:.1f}) score={score:.2f}"
                )
                changed = True

    if len(translations) != len(geoms):
        missing = sorted(set(geoms) - set(translations))
        raise RuntimeError(f"KNOWN6 profile layout konnte nicht alle Teile platzieren, missing={missing}")

    # Stabilisieren: gemeinsame Spalten und Reihen.
    top_ids = [int(x) for x in grid[0]]
    bottom_ids = [int(x) for x in grid[1]]

    for c in range(cols):
        ids_here = [top_ids[c], bottom_ids[c]]
        avg_col = float(np.mean([translations[pid][1] for pid in ids_here]))
        for pid in ids_here:
            translations[pid][1] = avg_col

    top_row = float(np.mean([translations[pid][0] for pid in top_ids]))
    bottom_row = float(np.mean([translations[pid][0] for pid in bottom_ids]))

    if bottom_row <= top_row:
        max_top_h = max(float(geoms[pid]["height"]) for pid in top_ids)
        bottom_row = top_row + max_top_h * 0.72

    for pid in top_ids:
        translations[pid][0] = top_row
    for pid in bottom_ids:
        translations[pid][0] = bottom_row

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

    for pid in sorted(translations):
        t = translations[pid]
        print(f"[KNOWN6 PROFILE FINAL] P{pid} translation=({t[0]:.1f},{t[1]:.1f})")

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
text = text[:start] + layout_block.strip() + "\n\n" + text[end:]

# Debugmeldung.
text = text.replace(
    'print("[KNOWN6 MODE] edge translation disabled -> fixed 2x3 cell layout")',
    'print("[KNOWN6 MODE] mask/profile matched layout")',
)
text = text.replace(
    'print("[KNOWN6 MODE] axis-snap compact 2x3 layout")',
    'print("[KNOWN6 MODE] mask/profile matched layout")',
)
if "[KNOWN6 MODE] mask/profile matched layout" not in text:
    marker = 'print(f"[KNOWN6 GRID] expected/current grid = {KNOWN_6PIECE_GRID}")'
    if marker in text:
        text = text.replace(
            marker,
            marker + '\n    print("[KNOWN6 MODE] mask/profile matched layout")',
            1,
        )

path.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(path), doraise=True)
except Exception as exc:
    path.write_text(orig, encoding="utf-8")
    raise RuntimeError(f"Patch erzeugte Syntaxfehler. Datei wurde zurueckgesetzt: {exc}")

print("[OK] Known6 Mask/Profile Layout v2 installiert.")
print(f"[BACKUP] {backup}")
print("")
print("Jetzt testen:")
print("    python tests/run_robot_solver_small.py")
print("")
print("Erwartete Konsolenausgabe:")
print("    [KNOWN6 MODE] mask/profile matched layout")
print("    [KNOWN6 ROTMASK] Piece ...")
print("    [KNOWN6 PROFILE] ...")
print("")
print("Feintuning bei leichter Ueberlappung:")
print("    KNOWN_6PIECE_PROFILE_CLEARANCE_PX = 2.0 oder 4.0")
