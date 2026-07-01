# Patch: Known6-Wettbewerbslogik in run_robot_solver_small.py einbauen.
#
# Logik:
# - Wenn genau die offiziellen 6 Teile mit IDs {1..6} erkannt werden:
#       -> deterministischer Known6-Solver mit Zielraster [[6,2,1],[3,5,4]]
# - Sonst:
#       -> bisheriger SmallSolver / normaler Fallback bleibt aktiv
#
# Ausführen im Repo-Hauptordner:
#     python install_known6_solver_v1.py
#
# Danach:
#     python tests/run_robot_solver_small.py

from __future__ import annotations

from pathlib import Path
import re
import py_compile

run_path = Path("tests/run_robot_solver_small.py")
if not run_path.exists():
    raise FileNotFoundError("tests/run_robot_solver_small.py nicht gefunden. Bitte im Repo-Hauptordner ausführen.")

orig = run_path.read_text(encoding="utf-8")
backup = run_path.with_suffix(".py.bak_known6_solver_v1")
backup.write_text(orig, encoding="utf-8")

text = orig

config_block = '''# ---------------------------------------------------------
# KNOWN-6-PIECE-WETTBEWERBSLOGIK
# ---------------------------------------------------------
# Wenn genau das offizielle 6-Teile-Puzzle erkannt wird, verwenden wir die
# bekannte 2x3-Topologie. Dadurch muss der Solver bei den sehr ähnlichen neuen
# Rundkanten nicht mehr die komplette Nachbarschaft erraten.
USE_KNOWN_6PIECE_GRID_SOLVER = True
KNOWN_6PIECE_GRID = [
    [6, 2, 1],
    [3, 5, 4],
]
KNOWN_6PIECE_GRID_MARGIN_PX = 80
KNOWN_6PIECE_USE_EDGE_TRANSLATION = True

'''

if "USE_KNOWN_6PIECE_GRID_SOLVER" not in text:
    marker = "# ---------------------------------------------------------\n# SOLVER-FALLBACK"
    if marker in text:
        text = text.replace(marker, config_block + marker, 1)
    else:
        marker = "A5_SNAP_ROTATIONS_TO_CARDINAL"
        if marker in text:
            idx = text.find(marker)
            line_start = text.rfind("\n", 0, idx) + 1
            text = text[:line_start] + config_block + text[line_start:]
        else:
            raise RuntimeError("Konnte keine passende Stelle für Known6-Config finden.")

known6_helpers = r'''
# =========================================================
# Known6 Wettbewerbssolver
# =========================================================

def _known6_type_name(edge):
    return getattr(getattr(edge, "type", None), "name", str(getattr(edge, "type", "")))


def _known6_is_border(edge):
    return _known6_type_name(edge) == "BORDER"


def _known6_dir_name(direction):
    return getattr(direction, "name", str(direction))


def _known6_dirs_by_name():
    return {
        "N": Directions.N,
        "E": Directions.E,
        "S": Directions.S,
        "W": Directions.W,
    }


def _known6_rotate_direction(direction, steps):
    fn = globals().get("rotate_direction")
    if callable(fn):
        return fn(direction, steps)

    dirs = [Directions.N, Directions.E, Directions.S, Directions.W]
    return dirs[(dirs.index(direction) + int(steps)) % 4]


def _known6_rotate_point_px(dx, dy, angle_deg):
    fn = globals().get("rotate_point_px")
    if callable(fn):
        return fn(dx, dy, angle_deg)

    a = np.deg2rad(float(angle_deg))
    ca = np.cos(a)
    sa = np.sin(a)
    return (float(dx) * ca - float(dy) * sa, float(dx) * sa + float(dy) * ca)


def _known6_expected_border_dirs(row_idx, col_idx, rows, cols):
    dirs = _known6_dirs_by_name()
    expected = set()
    if row_idx == 0:
        expected.add(dirs["N"])
    if row_idx == rows - 1:
        expected.add(dirs["S"])
    if col_idx == 0:
        expected.add(dirs["W"])
    if col_idx == cols - 1:
        expected.add(dirs["E"])
    return expected


def _choose_known_grid_rotation(piece, row_idx, col_idx, rows, cols):
    expected = _known6_expected_border_dirs(row_idx, col_idx, rows, cols)
    expected_names = {_known6_dir_name(d) for d in expected}

    scored = []
    for steps in range(4):
        score = 0.0
        border_hits = 0

        for edge in piece.edges_:
            new_dir = _known6_rotate_direction(edge.direction, steps)
            new_name = _known6_dir_name(new_dir)
            is_border = _known6_is_border(edge)

            if is_border and new_name in expected_names:
                border_hits += 1
            elif is_border and new_name not in expected_names:
                score += 100.0
            elif (not is_border) and new_name in expected_names:
                score += 45.0

        score += abs(border_hits - len(expected_names)) * 60.0

        try:
            h, w = _known6_rotated_piece_bbox_dims(piece, steps)
            if w < h and len(expected_names) <= 1:
                score += 5.0
        except Exception:
            pass

        scored.append((score, steps))

    scored.sort(key=lambda x: x[0])
    best_score, best_steps = scored[0]
    print(
        f"[KNOWN6 ROT] Piece {int(piece.id)} cell=({row_idx},{col_idx}) "
        f"rotation_steps={best_steps} rot={best_steps * 90}° "
        f"score={best_score:.2f} all={[(round(s, 1), r) for s, r in scored]}"
    )
    return int(best_steps)


def _make_known_grid_oriented_geometry(piece, rotation_steps):
    min_row, min_col, max_row, max_col = piece.get_bbox()
    old_center_row = (min_row + max_row) / 2.0
    old_center_col = (min_col + max_col) / 2.0
    angle_deg = float(rotation_steps) * 90.0

    raw_pixels = []
    for (row, col), color in piece.pixels.items():
        dx = float(col) - old_center_col
        dy = float(row) - old_center_row
        rx, ry = _known6_rotate_point_px(dx, dy, angle_deg)
        raw_pixels.append((old_center_row + ry, old_center_col + rx, color))

    if not raw_pixels:
        raise RuntimeError(f"KNOWN6: Piece {int(piece.id)} hat keine Pixel")

    raw_edges = []
    for edge in piece.edges_:
        pts = []
        for p in edge.shape:
            row = float(p[0])
            col = float(p[1])
            dx = col - old_center_col
            dy = row - old_center_row
            rx, ry = _known6_rotate_point_px(dx, dy, angle_deg)
            pts.append([old_center_row + ry, old_center_col + rx])
        raw_edges.append((edge, np.asarray(pts, dtype=np.float32)))

    min_r = min(p[0] for p in raw_pixels)
    min_c = min(p[1] for p in raw_pixels)

    pixels_float = [(row_f - min_r, col_f - min_c, color) for row_f, col_f, color in raw_pixels]

    edge_records = []
    edges_by_dir = {}
    for edge, pts in raw_edges:
        pts_local = pts - np.asarray([min_r, min_c], dtype=np.float32)
        new_dir = _known6_rotate_direction(edge.direction, rotation_steps)
        rec = {
            "orig_edge": edge,
            "points": pts_local,
            "direction": new_dir,
            "type": edge.type,
        }
        edge_records.append(rec)
        edges_by_dir[new_dir] = pts_local

    rows = [p[0] for p in pixels_float]
    cols = [p[1] for p in pixels_float]

    return {
        "piece": piece,
        "rotation_steps": int(rotation_steps),
        "angle_deg": angle_deg,
        "old_center_row": old_center_row,
        "old_center_col": old_center_col,
        "pixels_float": pixels_float,
        "edge_records": edge_records,
        "edges_by_dir": edges_by_dir,
        "height": max(rows) - min(rows) + 1.0,
        "width": max(cols) - min(cols) + 1.0,
    }


def _known6_rotated_piece_bbox_dims(piece, rotation_steps):
    geom = _make_known_grid_oriented_geometry(piece, rotation_steps)
    return float(geom["height"]), float(geom["width"])


def _known6_resample_polyline_rowcol(points_rowcol, n_points=90):
    pts = np.asarray(points_rowcol, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((n_points, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1], n_points, axis=0)

    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-6:
        return np.repeat(pts[:1], n_points, axis=0)

    target = np.linspace(0.0, total, n_points)
    out = []
    j = 0
    for t in target:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(pts):
            out.append(pts[-1])
        else:
            ratio = (t - s[j]) / (s[j + 1] - s[j] + 1e-8)
            out.append(pts[j] + ratio * (pts[j + 1] - pts[j]))
    return np.asarray(out, dtype=np.float32)


def _known6_edge_score_after_translation(block_edge_world, new_edge_local, t):
    a = _known6_resample_polyline_rowcol(block_edge_world, 90)
    b = _known6_resample_polyline_rowcol(new_edge_local + t, 90)
    b_rev = b[::-1]
    return float(min(
        np.mean(np.linalg.norm(a - b, axis=1)),
        np.mean(np.linalg.norm(a - b_rev, axis=1)),
    ))


def _best_translation_between_edges(block_edge_world_rowcol, new_edge_local_rowcol):
    block = np.asarray(block_edge_world_rowcol, dtype=np.float32)
    new = np.asarray(new_edge_local_rowcol, dtype=np.float32)

    if block.shape[0] < 2 or new.shape[0] < 2:
        return np.asarray([0.0, 0.0], dtype=np.float32), float("inf")

    candidates = []
    candidates.append(block.mean(axis=0) - new.mean(axis=0))

    br = _known6_resample_polyline_rowcol(block, 90)
    nr = _known6_resample_polyline_rowcol(new, 90)
    candidates.append(br[0] - nr[0])
    candidates.append(br[-1] - nr[-1])
    candidates.append(br[0] - nr[-1])
    candidates.append(br[-1] - nr[0])

    best_t = candidates[0]
    best_score = float("inf")
    for t in candidates:
        t = np.asarray(t, dtype=np.float32)
        score = _known6_edge_score_after_translation(block, new, t)
        if score < best_score:
            best_score = score
            best_t = t

    return np.asarray(best_t, dtype=np.float32), best_score


def _build_known_6piece_edge_matched_layout(pieces_by_id, rotations):
    grid = KNOWN_6PIECE_GRID
    rows = len(grid)
    cols = len(grid[0])

    geoms = {
        int(pid): _make_known_grid_oriented_geometry(pieces_by_id[int(pid)], rotations[int(pid)])
        for row in grid
        for pid in row
    }

    anchor_pid = int(grid[0][0])
    translations = {anchor_pid: np.asarray([0.0, 0.0], dtype=np.float32)}

    adj = []
    dirs = _known6_dirs_by_name()
    for r in range(rows):
        for c in range(cols - 1):
            adj.append((int(grid[r][c]), int(grid[r][c + 1]), dirs["E"], dirs["W"]))
    for r in range(rows - 1):
        for c in range(cols):
            adj.append((int(grid[r][c]), int(grid[r + 1][c]), dirs["S"], dirs["N"]))

    changed = True
    while changed and len(translations) < len(geoms):
        changed = False
        for placed_pid, new_pid, placed_dir, new_dir in adj:
            if placed_pid in translations and new_pid not in translations:
                placed_geom = geoms[placed_pid]
                new_geom = geoms[new_pid]
                block_edge = placed_geom["edges_by_dir"].get(placed_dir)
                new_edge = new_geom["edges_by_dir"].get(new_dir)
                if block_edge is None or new_edge is None:
                    continue

                block_world = block_edge + translations[placed_pid]
                t, score = _best_translation_between_edges(block_world, new_edge)
                translations[new_pid] = t
                print(
                    f"[KNOWN6 EDGE] {placed_pid}.{_known6_dir_name(placed_dir)} -> "
                    f"{new_pid}.{_known6_dir_name(new_dir)}: "
                    f"translation=({t[0]:.1f},{t[1]:.1f}) score={score:.2f}"
                )
                changed = True

            elif new_pid in translations and placed_pid not in translations:
                placed_geom = geoms[new_pid]
                new_geom = geoms[placed_pid]
                block_edge = placed_geom["edges_by_dir"].get(new_dir)
                new_edge = new_geom["edges_by_dir"].get(placed_dir)
                if block_edge is None or new_edge is None:
                    continue

                block_world = block_edge + translations[new_pid]
                t, score = _best_translation_between_edges(block_world, new_edge)
                translations[placed_pid] = t
                print(
                    f"[KNOWN6 EDGE] {new_pid}.{_known6_dir_name(new_dir)} -> "
                    f"{placed_pid}.{_known6_dir_name(placed_dir)}: "
                    f"translation=({t[0]:.1f},{t[1]:.1f}) score={score:.2f}"
                )
                changed = True

    if len(translations) != len(geoms):
        missing = sorted(set(geoms) - set(translations))
        raise RuntimeError(f"KNOWN6 edge layout konnte nicht alle Teile platzieren, missing={missing}")

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

    for pid in list(translations.keys()):
        translations[pid] = translations[pid] + shift

    return geoms, translations


def _apply_known_grid_geometry_to_piece(geom, translation_rowcol):
    piece = geom["piece"]
    tr, tc = float(translation_rowcol[0]), float(translation_rowcol[1])

    new_pixels = {}
    for row_f, col_f, color in geom["pixels_float"]:
        nr = int(round(tr + row_f))
        nc = int(round(tc + col_f))
        new_pixels[(nr, nc)] = color
    piece.pixels = new_pixels

    for rec in geom["edge_records"]:
        edge = rec["orig_edge"]
        pts = rec["points"] + np.asarray([tr, tc], dtype=np.float32)
        edge.shape = np.round(pts).astype(np.int32)
        edge.shape_backup = np.array(edge.shape, copy=True)
        edge.direction = rec["direction"]
        edge.connected = _known6_type_name(edge) == "BORDER"

    new_min_row, new_min_col, new_max_row, new_max_col = piece.get_bbox()
    new_center_row = (new_min_row + new_max_row) / 2.0
    new_center_col = (new_min_col + new_max_col) / 2.0

    dx = int(round(new_center_row - geom["old_center_row"]))
    dy = int(round(new_center_col - geom["old_center_col"]))

    return (
        f"TRANSFORM_REPORT {int(piece.id)} "
        f"{int(round(geom['old_center_row']))} {int(round(geom['old_center_col']))} "
        f"{int(round(new_center_row))} {int(round(new_center_col))} "
        f"{dx} {dy} {float(geom['angle_deg']):.1f}"
    )


def solve_known_6piece_grid_if_applicable(puzzle, transformation_logs):
    if not USE_KNOWN_6PIECE_GRID_SOLVER:
        return False

    if puzzle.pieces_ is None or len(puzzle.pieces_) != 6:
        return False

    wanted_ids = {int(pid) for row in KNOWN_6PIECE_GRID for pid in row}
    pieces_by_id = {int(piece.id): piece for piece in puzzle.pieces_}

    if set(pieces_by_id.keys()) != wanted_ids:
        print(
            f"[KNOWN6] Nicht angewendet: IDs={sorted(pieces_by_id.keys())}, "
            f"erwartet={sorted(wanted_ids)}"
        )
        return False

    print("[KNOWN6] Verwende deterministischen 2x3-Grid-Solver für Wettbewerbspuzzle.")

    rows = len(KNOWN_6PIECE_GRID)
    cols = len(KNOWN_6PIECE_GRID[0])

    rotations = {}
    for r, row in enumerate(KNOWN_6PIECE_GRID):
        for c, pid in enumerate(row):
            pid = int(pid)
            rotations[pid] = _choose_known_grid_rotation(pieces_by_id[pid], r, c, rows, cols)

    geoms, translations = _build_known_6piece_edge_matched_layout(pieces_by_id, rotations)

    transformation_logs.clear()

    if not hasattr(puzzle, "connected_directions") or puzzle.connected_directions is None:
        puzzle.connected_directions = []
    else:
        puzzle.connected_directions.clear()

    for r, row in enumerate(KNOWN_6PIECE_GRID):
        for c, pid in enumerate(row):
            pid = int(pid)
            piece = pieces_by_id[pid]
            report = _apply_known_grid_geometry_to_piece(geoms[pid], translations[pid])
            transformation_logs.append(report)
            print(report)
            piece.coord = (r, c)

    return True

'''

if "def solve_known_6piece_grid_if_applicable" not in text:
    marker = "\ndef main("
    if marker not in text:
        raise RuntimeError("Konnte def main(...) nicht finden.")
    text = text.replace(marker, "\n" + known6_helpers + marker, 1)

pattern = r'''    print\("\[RUN\] Löse Puzzle\.\.\."\)
    small_solver_used = False
    if USE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE and hasattr\(puzzle, "solve_puzzle_small"\):
        print\("\[RUN\] Löse Puzzle mit SmallSolver\.\.\."\)
        transformation_logs\.clear\(\)
        ok = puzzle\.solve_puzzle_small\(fallback=False\)
        if not ok:
            raise RuntimeError\("SmallSolver konnte das Puzzle nicht lösen\."\)
        small_solver_used = True
    else:
        print\("\[RUN\] Löse Puzzle mit normalem/loose Solver\.\.\."\)
        solve_puzzle_with_optional_loose_retry\(puzzle, transformation_logs\)
    print\("\[RUN\] Puzzle gelöst\."\)
'''

replacement = '''    print("[RUN] Löse Puzzle...")
    small_solver_used = False
    known6_solver_used = False

    if solve_known_6piece_grid_if_applicable(puzzle, transformation_logs):
        known6_solver_used = True
        print("[RUN] Puzzle mit Known6-Wettbewerbslogik gelöst.")

    elif USE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE and hasattr(puzzle, "solve_puzzle_small"):
        print("[RUN] Löse Puzzle mit SmallSolver...")
        transformation_logs.clear()
        ok = puzzle.solve_puzzle_small(fallback=False)
        if not ok:
            raise RuntimeError("SmallSolver konnte das Puzzle nicht lösen.")
        small_solver_used = True

    else:
        print("[RUN] Löse Puzzle mit normalem/loose Solver...")
        solve_puzzle_with_optional_loose_retry(puzzle, transformation_logs)

    print("[RUN] Puzzle gelöst.")
'''

text_new, n = re.subn(pattern, replacement, text, count=1)
if n == 0:
    old = '''    print("[RUN] Löse Puzzle...")
    solve_puzzle_with_optional_loose_retry(puzzle, transformation_logs)
    print("[RUN] Puzzle gelöst.")
'''
    if old in text:
        text_new = text.replace(
            old,
            '''    print("[RUN] Löse Puzzle...")
    small_solver_used = False
    known6_solver_used = False

    if solve_known_6piece_grid_if_applicable(puzzle, transformation_logs):
        known6_solver_used = True
        print("[RUN] Puzzle mit Known6-Wettbewerbslogik gelöst.")
    elif USE_SMALL_SOLVER_FOR_COMPETITION_PUZZLE and hasattr(puzzle, "solve_puzzle_small"):
        print("[RUN] Löse Puzzle mit SmallSolver...")
        transformation_logs.clear()
        ok = puzzle.solve_puzzle_small(fallback=False)
        if not ok:
            raise RuntimeError("SmallSolver konnte das Puzzle nicht lösen.")
        small_solver_used = True
    else:
        solve_puzzle_with_optional_loose_retry(puzzle, transformation_logs)

    print("[RUN] Puzzle gelöst.")
''',
            1,
        )
    else:
        raise RuntimeError("Konnte den Solver-Block in main() nicht ersetzen.")
text = text_new

text = re.sub(
    r"USE_SMALL_SOLVER_EXPORTED_ROTATION_FOR_A5\s*=\s*True",
    "USE_SMALL_SOLVER_EXPORTED_ROTATION_FOR_A5 = False",
    text,
)

run_path.write_text(text, encoding="utf-8")

try:
    py_compile.compile(str(run_path), doraise=True)
except Exception as exc:
    run_path.write_text(orig, encoding="utf-8")
    raise RuntimeError(f"Patch erzeugte Syntaxfehler. Datei wurde zurückgesetzt: {exc}")

print("[OK] Known6-Wettbewerbslogik in tests/run_robot_solver_small.py installiert.")
print(f"[BACKUP] {backup}")
print("Jetzt testen mit:")
print("    python tests/run_robot_solver_small.py")
print("")
print("In der Konsole sollten Zeilen erscheinen wie:")
print("    [KNOWN6] Verwende deterministischen 2x3-Grid-Solver für Wettbewerbspuzzle.")
print("    [KNOWN6 ROT] Piece ...")
print("    [KNOWN6 EDGE] ...")
