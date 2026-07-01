"""Dedicated solver for small puzzles (<=6 pieces, always a 3x2 or 2x3 grid).

Such a puzzle is exactly **4 corner pieces + 2 edge pieces**. Instead of forcing pieces
into a fixed slot template (which can commit a bad edge), this assembles by **confidence**:
score every candidate edge pair once, then merge pieces **best-first** (agglomerative).

The search is purely combinatorial on a **grid embedding** — each piece is tracked as a grid
cell + orientation; a seam fixes the relative cell/orientation of the two clusters. A merge is
accepted only if it stays grid-legal (no cell collision; 3x2/2x3 footprint; no BORDER edge
facing another piece). Best-first DFS with rollback recovers from dead-ends, so poor seams are
used last or never. Geometry is materialised once at the end (stick along grid adjacency) for
rendering / robot coords. The 4-corner+2-edge structure is the validation guard, not a fixed
placement order.
"""
import os
import numpy as np
import math
import cv2

from .. import config
from .Distance import _geom_match_score, _offset_shape_for
from .Mover import stick_pieces
from .Enums import TypeEdge


# --------------------------------------------------------------------------- #
# Geometry helpers (shape coords are (col=x, row=y); row grows downward)
# --------------------------------------------------------------------------- #

def _ssr_norm_deg(angle):
    # Normalize angle to [-180, 180).
    a = float(angle)
    while a < -180.0:
        a += 360.0
    while a >= 180.0:
        a -= 360.0
    return a

def _centroid(piece):
    pts = [np.asarray(e.shape, dtype=float) for e in piece.edges_ if len(e.shape) > 0]
    return np.concatenate(pts, 0).mean(0)




def _apply_rigid_to_pixels(piece, R, center_xy, translation_xy=(0.0, 0.0)):
    """Apply the same solved-layout transform to piece.pixels as to edge.shape.

    edge.shape uses (x=col, y=row), while piece.pixels keys are (row, col).
    Therefore we convert every pixel key to (col,row), transform it, and store
    it back as (row,col). Without this, A5 placement/debug still uses the old
    unsolved source image positions.
    """
    if not getattr(piece, "pixels", None):
        return

    c = np.asarray(center_xy, dtype=float)
    t = np.asarray(translation_xy, dtype=float)
    new_pixels = {}

    for (row, col), color in piece.pixels.items():
        p_xy = np.array([float(col), float(row)], dtype=float)
        q_xy = (p_xy - c) @ R.T + c + t
        new_col = int(round(q_xy[0]))
        new_row = int(round(q_xy[1]))
        new_pixels[(new_row, new_col)] = color

    piece.pixels = new_pixels


def _translate_pixels_xy(piece, translation_xy):
    """Translate pixels by an (x=col, y=row) vector."""
    tx, ty = float(translation_xy[0]), float(translation_xy[1])
    if not getattr(piece, "pixels", None):
        return
    piece.pixels = {
        (int(round(row + ty)), int(round(col + tx))): color
        for (row, col), color in piece.pixels.items()
    }


def _snapshot(pieces):
    return {id(e): np.copy(e.shape) for p in pieces for e in p.edges_}


def _restore(pieces, snap):
    for p in pieces:
        for e in p.edges_:
            e.shape = np.copy(snap[id(e)])


def _connector_apex_depth(pts):
    """Perpendicular distance from the connector apex to the chord (HEAD=positive, HOLE=negative)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0
    a, b = pts[0], pts[-1]
    ch = b - a
    L = np.linalg.norm(ch)
    if L < 1e-6:
        return 0.0
    u = ch / L
    nrm = np.array([-u[1], u[0]])
    devs = (pts - a) @ nrm
    idx = int(np.argmax(np.abs(devs)))
    return float(devs[idx])


def _connector_midpoint_fraction(pts):
    """Arc-length fraction (0..1) along the edge where the connector apex sits."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.5
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    total = cum[-1]
    if total < 1e-6:
        return 0.5
    a, b = pts[0], pts[-1]
    ch = b - a
    L = np.linalg.norm(ch)
    if L < 1e-6:
        return 0.5
    u = ch / L
    nrm = np.array([-u[1], u[0]])
    devs = np.abs((pts - a) @ nrm)
    idx = int(np.argmax(devs))
    return float(cum[idx] / total)


def _seam_cost(cand_edge, bloc_edge, cand, bloc, green):
    """Shape-match cost for a seam. Uses _geom_match_score (overlap residual +
    curvature + type priority) plus two additional discriminators:
    - connector apex depth match (HEAD depth ≈ -HOLE depth)
    - connector midpoint position along the edge
    These extra terms prevent the DFS from accepting wrong piece assignments
    when all connector shapes look similar."""
    s1 = _offset_shape_for(cand_edge, _centroid(cand))
    s2 = _offset_shape_for(bloc_edge, _centroid(bloc))
    base = _geom_match_score(s1, s2, cand_edge, bloc_edge)

    # Extra discriminators for any non-BORDER pair, including UNDEFINED edges
    # (upstream classification sometimes fails to label a real HEAD/HOLE edge).
    # Apex depth/midpoint position are properties of the edge's own geometry —
    # available and meaningful even when Edge.type couldn't be determined
    # upstream, so we apply them generically rather than only for typed pairs.
    # NOTE: depth sign is relative to each piece's OWN centroid, so a true
    # matching HEAD/HOLE pair does not reliably have opposite signs here —
    # only the magnitude is comparable (see _connector_apex_depth docstring).
    from .Enums import TypeEdge
    if cand_edge.type != TypeEdge.BORDER and bloc_edge.type != TypeEdge.BORDER:
        d1 = _connector_apex_depth(s1)
        d2 = _connector_apex_depth(s2)
        # Depth magnitude should match for a true mating pair. Penalise mismatch.
        depth_penalty = abs(abs(d1) - abs(d2)) * 0.5
        # Connector apex should sit at the same fractional position along both edges.
        f1 = _connector_midpoint_fraction(s1)
        f2 = _connector_midpoint_fraction(s2)
        pos_penalty = abs(f1 - f2) * 300.0
        base += depth_penalty + pos_penalty

    return base


# --------------------------------------------------------------------------- #
# Grid directions. edges_ run in the fixed cyclic order below (verified: opposite =
# (k+2)%4); a piece's "offset" o maps edge index k -> grid direction _CYCLE[(k+o)%4].
# --------------------------------------------------------------------------- #
_CYCLE = ['S', 'E', 'N', 'W']
_CYC_IX = {d: i for i, d in enumerate(_CYCLE)}
_OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
_STEP = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}   # (drow, dcol)


def _dir_of(edge_idx, off):
    return _CYCLE[(edge_idx + off) % 4]


def _rot_cell(vec, k):
    """Rotate a (drow, dcol) grid vector by k +1-steps; one step: (dr,dc)->(-dc,dr)."""
    dr, dc = vec
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return (dr, dc)


# --------------------------------------------------------------------------- #
# Seam scoring
# --------------------------------------------------------------------------- #
def _precompute_seams(pieces, green):
    """Score every compatible edge pair once (pose-independent: stick into a scratch frame,
    score, restore). Returns records {cost, pi, ki, pj, kj} (k* = edge index) best-first."""
    seams = []
    n = len(pieces)
    for i in range(n):
        for j in range(i + 1, n):
            pair = [pieces[i], pieces[j]]
            for ki, ei in enumerate(pieces[i].edges_):
                for kj, ej in enumerate(pieces[j].edges_):
                    if not ei.is_compatible(ej):
                        continue
                    snap = _snapshot(pair)
                    stick_pieces(ei, pieces[j], ej,
                                 centroid_bloc=_centroid(pieces[i]),
                                 centroid_cand=_centroid(pieces[j]))
                    cost = _seam_cost(ej, ei, pieces[j], pieces[i], green)
                    _restore(pair, snap)
                    if np.isfinite(cost):
                        seams.append({'cost': cost, 'pi': i, 'ki': ki, 'pj': j, 'kj': kj})
    seams.sort(key=lambda s: s['cost'])
    return seams


# --------------------------------------------------------------------------- #
# Combinatorial grid embedding (union-find clusters of grid cells + orientations)
# --------------------------------------------------------------------------- #
def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _candidate_dims(n, nC, nE, nX):
    """Rectangle dimensions (w<=h, both >=2) of n pieces whose expected corner/edge/center
    counts match the actual classification. A w x h grid has 4 corners, 2(w-2)+2(h-2) edges,
    (w-2)(h-2) centers. Returns sorted (w,h) tuples (e.g. 4 -> [(2,2)], 6 -> [(2,3)])."""
    out = []
    for w in range(2, int(n ** 0.5) + 1):
        if n % w:
            continue
        h = n // w
        if (4, 2 * (w - 2) + 2 * (h - 2), (w - 2) * (h - 2)) == (nC, nE, nX):
            out.append((w, h))
    return out


def _fits_any(er, ec, cands):
    """Does a partial bbox of extents (er, ec) still fit inside some candidate rectangle?"""
    return any((er <= w and ec <= h) or (er <= h and ec <= w) for (w, h) in cands)


def _grid_merge(seam, pieces, parent, members, cell, off, cands):
    """Embed the smaller cluster into the larger's grid frame via this seam. Accept only if
    grid-legal: no cell collision, bbox still fits a candidate rectangle, and no BORDER edge
    facing an occupied cell. Mutates (cell/off/parent/members), returns True on accept."""
    ri, rj = _uf_find(parent, seam['pi']), _uf_find(parent, seam['pj'])
    if ri == rj:
        return False
    if len(members[ri]) >= len(members[rj]):
        a_i, a_k, b_i, b_k, fixed_root, move_root = \
            seam['pi'], seam['ki'], seam['pj'], seam['kj'], ri, rj
    else:
        a_i, a_k, b_i, b_k, fixed_root, move_root = \
            seam['pj'], seam['kj'], seam['pi'], seam['ki'], rj, ri

    d_a = _dir_of(a_k, off[a_i])                       # direction from A toward B
    rB = (_CYC_IX[_OPP[d_a]] - b_k) % 4                # B's edge b_k must face back
    cellB = (cell[a_i][0] + _STEP[d_a][0], cell[a_i][1] + _STEP[d_a][1])
    delta = (rB - off[b_i]) % 4

    occupied = {cell[k]: k for k in members[fixed_root]}
    new_cells, new_offs = {}, {}
    for q in members[move_root]:
        vec = _rot_cell((cell[q][0] - cell[b_i][0], cell[q][1] - cell[b_i][1]), delta)
        cq = (cellB[0] + vec[0], cellB[1] + vec[1])
        if cq in occupied:
            return False
        occupied[cq] = q
        new_cells[q], new_offs[q] = cq, (off[q] + delta) % 4

    rows = [c[0] for c in occupied]
    cols = [c[1] for c in occupied]
    er, ec = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    if not _fits_any(er, ec, cands):
        return False

    comb_off = {k: off[k] for k in members[fixed_root]}
    comb_off.update(new_offs)
    for cc, pidx in occupied.items():
        for k, e in enumerate(pieces[pidx].edges_):
            if e.type == TypeEdge.BORDER:
                d = _dir_of(k, comb_off[pidx])
                if (cc[0] + _STEP[d][0], cc[1] + _STEP[d][1]) in occupied:
                    return False

    for q in members[move_root]:
        cell[q], off[q] = new_cells[q], new_offs[q]
    parent[move_root] = fixed_root
    members[fixed_root].extend(members[move_root])
    del members[move_root]
    return True


def _snap_grid(parent, members, cell, off):
    return (list(parent), {k: list(v) for k, v in members.items()}, dict(cell), dict(off))


def _restore_grid(snap, parent, members, cell, off):
    parent[:] = snap[0]
    members.clear(); members.update({k: list(v) for k, v in snap[1].items()})
    cell.clear(); cell.update(snap[2])
    off.clear(); off.update(snap[3])


def _structure_ok(pieces, cell, off, cands):
    """A candidate-rectangle footprint where every CONNECTOR edge faces an occupied
    neighbour and every BORDER edge faces empty space (corners at grid corners, edges on
    the hull, centers interior)."""
    n = len(pieces)
    rows = [cell[i][0] for i in range(n)]
    cols = [cell[i][1] for i in range(n)]
    er, ec = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    if (min(er, ec), max(er, ec)) not in cands:
        return False
    occ = {cell[i] for i in range(n)}
    for i, p in enumerate(pieces):
        for k, e in enumerate(p.edges_):
            d = _dir_of(k, off[i])
            has = (cell[i][0] + _STEP[d][0], cell[i][1] + _STEP[d][1]) in occ
            if e.type == TypeEdge.BORDER and has:
                return False
            # UNDEFINED edges are treated as connectors here: an edge piece
            # placed at a corner cell could otherwise "pass" because only its
            # one BORDER edge is checked while the remaining UNDEFINED edges
            # (which are really connectors) go unchecked even when facing empty
            # exterior cells.  Treating UNDEFINED as connector closes that hole.
            if e.type in (TypeEdge.HEAD, TypeEdge.HOLE, TypeEdge.UNDEFINED) and not has:
                return False
    return True


def _dfs(seams, si, pieces, parent, members, cell, off, acc_cost, budget, best, cands):
    """Search structurally-valid complete assemblies, keeping the MIN-TOTAL-COST one in
    `best`. Per-seam shape scores can't separate true/false matches for these pieces, so
    we let the scores decide globally: a locally-cheap false pairing can't win if it forces
    a higher-total or structurally-invalid whole. Branch-and-bound on accumulated cost."""
    if len(members) == 1:
        if _structure_ok(pieces, cell, off, cands):
            _cv = list(cell.values())
            _er = max(r[0] for r in _cv) - min(r[0] for r in _cv) + 1
            _ec = max(r[1] for r in _cv) - min(r[1] for r in _cv) + 1
            # The competition A5 surface is landscape (wider than tall). Prefer
            # assemblies where columns > rows (landscape grid). Portrait grids
            # get a penalty so the DFS only accepts them when they're significantly
            # cheaper — preventing false-positive 2×3 solutions when 3×2 is correct.
            portrait_penalty = 100.0 if _er > _ec else 0.0
            effective_cost = acc_cost + portrait_penalty
            if effective_cost < best['cost']:
                best['cost'] = effective_cost
                best['cell'] = dict(cell)
                best['off'] = dict(off)
        return
    if acc_cost >= best['cost']:        # bound: can't beat the incumbent
        return
    budget[0] -= 1
    if budget[0] < 0:
        return
    for idx in range(si, len(seams)):
        s = seams[idx]
        if _uf_find(parent, s['pi']) == _uf_find(parent, s['pj']):
            continue
        snap = _snap_grid(parent, members, cell, off)
        if _grid_merge(s, pieces, parent, members, cell, off, cands):
            _dfs(seams, idx + 1, pieces, parent, members, cell, off,
                 acc_cost + s['cost'], budget, best, cands)
            _restore_grid(snap, parent, members, cell, off)
    return


# --------------------------------------------------------------------------- #
# Materialise geometry from the grid embedding (axis-lock + least-squares place)
# --------------------------------------------------------------------------- #
_GLOBAL = {'N': np.array([0.0, -1.0]), 'S': np.array([0.0, 1.0]),
           'E': np.array([1.0, 0.0]), 'W': np.array([-1.0, 0.0])}   # outward (x, y), y down




def _resample_points_for_layout(points, n_points=80):
    """Resample polyline points to a fixed number of points along arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((n_points, 2), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts[:1], n_points, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return np.repeat(pts[:1], n_points, axis=0)

    target = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 2), dtype=float)
    j = 0
    for i, tt in enumerate(target):
        while j + 1 < len(s) and s[j + 1] < tt:
            j += 1
        if j + 1 >= len(pts):
            out[i] = pts[-1]
        else:
            denom = max(s[j + 1] - s[j], 1e-9)
            a = (tt - s[j]) / denom
            out[i] = pts[j] * (1.0 - a) + pts[j + 1] * a
    return out


def _edge_translation_delta(edge_i, edge_j, n_points=80):
    """Return translation t_j - t_i so edge_j fits edge_i.

    The pieces are already axis-locked. Remaining correction is mainly translation.
    We compare the complete opposing edge curves instead of only one apex point.
    """
    a = _resample_points_for_layout(edge_i.shape, n_points=n_points)
    b = _resample_points_for_layout(edge_j.shape, n_points=n_points)

    # Opposing edge polylines can be stored in opposite traversal direction.
    d_rev = a - b[::-1]
    d_fwd = a - b

    rev_spread = float(np.mean(np.linalg.norm(d_rev - np.mean(d_rev, axis=0), axis=1)))
    fwd_spread = float(np.mean(np.linalg.norm(d_fwd - np.mean(d_fwd, axis=0), axis=1)))
    d = d_rev if rev_spread <= fwd_spread else d_fwd
    return np.mean(d, axis=0)


def _apex_point(edge):
    """The connector apex (tab/hole centre) — point of max perpendicular deviation from
    the chord; the natural mating point of a seam, valid for any edge length (T-junctions)."""
    pts = np.asarray(edge.shape, dtype=float)
    a, b = pts[0], pts[-1]
    ch = b - a
    L = np.linalg.norm(ch)
    if L < 1e-6 or len(pts) < 3:
        return (a + b) / 2.0
    u = ch / L
    nrm = np.array([-u[1], u[0]])
    dev = (pts - a) @ nrm
    return pts[int(np.argmax(np.abs(dev)))]




# --- PATCH v5: force orthogonal/cardinal rotations for competition puzzle ---
def _ssv5_norm_angle_deg(a):
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def _ssv5_snap_delta_to_cardinal(angle_deg):
    """Return the smallest delta that rotates angle_deg to 0/90/180/-90."""
    target = round(angle_deg / 90.0) * 90.0
    return _ssv5_norm_angle_deg(target - angle_deg)


def _ssv5_piece_points_xy(piece):
    """Collect representative points of a piece as (x,y). Works with pixels and edge shapes."""
    pts = []
    pix = getattr(piece, 'pixels', None)
    if pix is not None:
        try:
            keys = pix.keys() if hasattr(pix, 'keys') else pix
            for q in keys:
                if len(q) >= 2:
                    # project code usually stores pixels as (x,y); if it is (row,col) this still only affects orientation modulo 90.
                    pts.append((float(q[0]), float(q[1])))
        except Exception:
            pass
    if len(pts) < 20:
        for e in getattr(piece, 'edges', []) or []:
            sh = getattr(e, 'shape', None)
            if sh is None:
                continue
            try:
                for q in sh:
                    pts.append((float(q[0]), float(q[1])))
            except Exception:
                pass
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def _ssv5_long_edge_orientation_deg(points):
    """Estimate the main rectangular axis from the minimum-area rectangle."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 5:
        return 0.0
    try:
        import cv2
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        best_len = -1.0
        best_ang = 0.0
        for i in range(4):
            v = box[(i + 1) % 4] - box[i]
            L = float(np.linalg.norm(v))
            if L > best_len:
                best_len = L
                best_ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
        # orientation modulo 90: only residual tilt matters
        while best_ang <= -45.0:
            best_ang += 90.0
        while best_ang > 45.0:
            best_ang -= 90.0
        return float(best_ang)
    except Exception:
        # PCA fallback
        c = pts.mean(axis=0)
        X = pts - c
        cov = X.T @ X
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmax(vals))]
        ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
        while ang <= -45.0:
            ang += 90.0
        while ang > 45.0:
            ang -= 90.0
        return float(ang)


def _ssv5_apply_rigid_to_piece(piece, angle_deg, center_xy):
    """Rotate all solver geometry of a piece around center_xy."""
    if abs(angle_deg) < 1e-9:
        return
    th = math.radians(angle_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=float)
    c = np.asarray(center_xy, dtype=float)

    # rotate edge shapes
    for e in getattr(piece, 'edges', []) or []:
        sh = getattr(e, 'shape', None)
        if sh is None:
            continue
        try:
            arr = np.asarray(sh, dtype=float)
            if arr.size == 0:
                continue
            arr2 = (arr - c) @ R.T + c
            if isinstance(sh, np.ndarray):
                e.shape = arr2
            else:
                e.shape = [tuple(map(float, q)) for q in arr2]
        except Exception:
            pass

    # rotate pixel dictionary / set / list
    pix = getattr(piece, 'pixels', None)
    if pix is not None:
        try:
            if hasattr(pix, 'items'):
                new_pix = {}
                for q, val in pix.items():
                    q2 = (np.asarray(q[:2], dtype=float) - c) @ R.T + c
                    new_pix[(int(round(q2[0])), int(round(q2[1])))] = val
                piece.pixels = new_pix
            else:
                new_pts = []
                for q in pix:
                    q2 = (np.asarray(q[:2], dtype=float) - c) @ R.T + c
                    new_pts.append((int(round(q2[0])), int(round(q2[1]))))
                try:
                    piece.pixels = type(pix)(new_pts)
                except Exception:
                    piece.pixels = new_pts
        except Exception:
            pass

    # rotate common center-like attrs if present
    for name in ('center', 'center_', 'centroid', 'centroid_', 'hole', 'hole_', 'screw_hole', 'screw_hole_', 'screw_hole_px'):
        if hasattr(piece, name):
            try:
                q = np.asarray(getattr(piece, name)[:2], dtype=float)
                q2 = (q - c) @ R.T + c
                setattr(piece, name, tuple(map(float, q2)))
            except Exception:
                pass


def _ssv5_force_piece_cardinal(piece, log=None):
    pts = _ssv5_piece_points_xy(piece)
    if len(pts) < 20:
        return 0.0
    angle = _ssv5_long_edge_orientation_deg(pts)
    delta = _ssv5_snap_delta_to_cardinal(angle)
    # Because angle is already residual in [-45,45], target is normally 0.
    # Ignore tiny numerical noise, otherwise rotate all geometry.
    if abs(delta) > 0.25:
        c = pts.mean(axis=0)
        _ssv5_apply_rigid_to_piece(piece, delta, c)
        if log:
            try:
                log(f'[small/cardinal] piece {getattr(piece, "id", getattr(piece, "id_", "?"))}: axis={angle:.2f}°, delta={delta:.2f}°')
            except Exception:
                pass
    return delta


def _ssv5_force_all_pieces_cardinal(pieces, log=None):
    for pc in pieces:
        _ssv5_force_piece_cardinal(pc, log=log)
# --- end PATCH v5 ---


def _axis_lock_all(pieces, off, log=print):
    """Rotate ALL pieces using a SINGLE shared reference angle plus each
    piece's EXACT off_i*90° grid offset, instead of each piece independently
    best-fitting its own rotation (the old per-piece _axis_lock).

    Why: independent per-piece fitting lets measurement noise on individual
    connector edges accumulate into a real rotation error that DIFFERS from
    piece to piece (observed up to ~10-25° per piece in practice). Even
    though the grid POSITIONS are exactly correct, this per-piece wobble
    means the assembled silhouette is not a true rectangle — visible as
    pieces overlapping their neighbours once placed close together.

    Since off_i is already known exactly from the solver (piece i's edges
    are SUPPOSED to face exactly _dir_of(k, off_i)), the only real unknown is
    the ONE shared reference angle that aligns the whole grid to the image.
    Fitting that once, by pooling every edge from every piece (each
    "un-rotated" by its own off_i*90° back to a common off=0 frame), and then
    applying ref + off_i*90 to each piece, makes every pair of pieces exactly
    90°*Δoff apart by construction — i.e. the result IS a rectangle, not
    just approximately one."""
    sin_sum = 0.0
    cos_sum = 0.0

    # Derivation: we want, per piece i and edge k,
    #     angle_of(cur_k) + a_i == target_angle(_CYCLE[(k+off_i)%4])
    # _CYCLE = ['S','E','N','W'] maps to target angles [90°, 0°, -90°, 180°] (see
    # _GLOBAL below) — each +1 step in cycle index is EXACTLY -90° in target-angle
    # space, so target_angle(_CYCLE[(k+off_i)%4]) == target_angle(_CYCLE[k]) - 90°*off_i,
    # giving:  a_i == target_angle(_CYCLE[k]) - angle_of(cur_k) - 90°*off_i.
    #
    # We then split a_i into a piece-independent shared part (ref_angle) plus
    # this piece's exact grid rotation. The split MUST be a_i = ref_angle - 90°*off_i
    # (minus, not plus) — only that choice makes ref_angle = target_angle(_CYCLE[k])
    # - angle_of(cur_k), i.e. independent of off_i, so that votes from pieces with
    # DIFFERENT off_i are estimating the SAME quantity and can be pooled together.
    # (An earlier version used "+90°*off_i" for the split, which leaves ref_angle
    # off_i-dependent — pooling votes across pieces with different off_i then
    # doesn't converge to a single correct value; verified empirically: it left a
    # 180° error on every odd-off_i piece while every even-off_i piece looked
    # fine, since a sign-flipped 90°*off_i term is wrong by 180°*off_i mod 360.)
    for i, p in enumerate(pieces):
        c = _centroid(p)
        for k, e in enumerate(p.edges_):
            pts = np.asarray(e.shape, dtype=float)
            if len(pts) == 0:
                continue
            cur = pts.mean(axis=0) - c
            n = np.linalg.norm(cur)
            if n < 1e-6:
                continue
            cur_angle = np.arctan2(cur[1], cur[0])
            tgt = _GLOBAL[_CYCLE[k % 4]]
            tgt_angle = np.arctan2(tgt[1], tgt[0])
            ang = tgt_angle - cur_angle
            sin_sum += np.sin(ang)
            cos_sum += np.cos(ang)

    ref_angle = (
        float(np.arctan2(sin_sum, cos_sum))
        if (abs(sin_sum) > 1e-12 or abs(cos_sum) > 1e-12)
        else 0.0
    )
    log(f"[small/axislock-global] shared reference angle = {np.degrees(ref_angle):.2f} deg")

    for i, p in enumerate(pieces):
        total_angle = ref_angle - np.deg2rad(90.0 * off[i])
        c = _centroid(p)

        prev_solver_rot = float(getattr(p, "small_solver_rotation_deg", 0.0))
        p.small_solver_rotation_deg = _ssr_norm_deg(
            prev_solver_rot + float(np.degrees(total_angle))
        )

        wrapped = ((total_angle + np.pi) % (2 * np.pi)) - np.pi
        if abs(wrapped) < np.deg2rad(0.15):
            continue

        R = np.array([
            [np.cos(total_angle), -np.sin(total_angle)],
            [np.sin(total_angle),  np.cos(total_angle)],
        ], dtype=float)

        for e in p.edges_:
            e.shape = np.round((np.asarray(e.shape, dtype=float) - c) @ R.T + c).astype(int)

        _apply_rigid_to_pixels(p, R, c)


def _check_solution_is_rectangle(pieces, min_fill_ratio=0.80, log=print):
    """Verify the combined solved-puzzle silhouette is (close to) a clean
    rectangle, via the convex hull's fill ratio inside its minAreaRect.

    A correct grid assembly with exact piece rotations should have its
    pieces' combined outline fill almost the entire minimum-area bounding
    rectangle (only the small notches at HOLE connectors on the outer
    boundary lose a little area). A low fill ratio means some piece is
    rotated/positioned wrong relative to the others even if the grid
    topology (which piece is where) was correct — e.g. a residual rotation
    error large enough to visibly tilt a piece out of the shared frame.

    Returns True if the assembly passes, False otherwise (logs either way)."""
    pts = []
    for p in pieces:
        for e in p.edges_:
            arr = np.asarray(e.shape, dtype=float)
            if len(arr):
                pts.extend(arr.tolist())

    if len(pts) < 20:
        log("[small/rect-check] not enough points to evaluate -> skipping check")
        return True

    pts_arr = np.asarray(pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts_arr)
    (_, _), (w, h), _ = rect
    rect_area = float(w) * float(h)

    hull = cv2.convexHull(pts_arr)
    hull_area = float(cv2.contourArea(hull))

    if rect_area < 1e-6:
        log("[small/rect-check] degenerate minAreaRect -> FAIL")
        return False

    fill_ratio = hull_area / rect_area
    ok = fill_ratio >= min_fill_ratio
    log(f"[small/rect-check] minAreaRect={w:.0f}x{h:.0f}px, "
        f"hull_fill_ratio={fill_ratio:.3f} (threshold={min_fill_ratio}) "
        f"{'OK' if ok else 'FAIL -> assembly is not a clean rectangle'}")
    return ok

def _piece_bbox_wh(piece):
    """Return (width, height) in (x=col, y=row) space from axis-locked edge shapes."""
    xs, ys = [], []
    for e in piece.edges_:
        arr = np.asarray(e.shape, dtype=float)
        if len(arr):
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
    if not xs:
        return 100.0, 100.0
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))


def _layout_overlap_ratio(pieces):
    """Return the fraction of piece pairs whose actual pixel masks overlap by
    more than a small tolerance. Centroid-distance heuristics are unreliable
    for L-shaped/concave puzzle pieces (two pieces can have far-apart centroids
    while their actual material still crosses) — real pixel masks (available
    on every piece as p.pixels) give a much more direct overlap signal."""
    masks = []
    for p in pieces:
        pixels = getattr(p, "pixels", None)
        if pixels:
            masks.append(set(pixels.keys()))
        else:
            masks.append(None)

    if all(m is not None for m in masks):
        n = len(pieces)
        bad = 0
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(masks[i] & masks[j])
                smaller = min(len(masks[i]), len(masks[j]))
                if smaller > 0 and inter / smaller > 0.05:
                    bad += 1
        pairs = n * (n - 1) // 2
        return bad / max(pairs, 1)

    # Fallback when pixel masks aren't available: centroid-distance heuristic.
    cents = []
    sizes = []
    for p in pieces:
        xs, ys = [], []
        for e in p.edges_:
            arr = np.asarray(e.shape, dtype=float)
            if len(arr):
                xs.extend(arr[:, 0].tolist())
                ys.extend(arr[:, 1].tolist())
        if xs:
            cents.append(np.array([np.mean(xs), np.mean(ys)]))
            sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        else:
            cents.append(np.zeros(2))
            sizes.append(100.0)
    n = len(pieces)
    bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(cents[i] - cents[j]))
            thresh = 0.30 * (sizes[i] + sizes[j]) / 2.0
            if dist < thresh:
                bad += 1
    pairs = n * (n - 1) // 2
    return bad / max(pairs, 1)


def _piece_core_bbox_wh(piece):
    """(width, height) using only each edge's corner endpoints, not its full
    curve. A HEAD tab bulges outward and a HOLE notch bulges inward along an
    edge's curve, but the polygon CORNERS (where adjacent edges meet — the
    first/last point of each edge) are unaffected by either, so this gives
    the piece's true core-rectangle size without tab-protrusion inflation."""
    corners = []
    for e in piece.edges_:
        pts = np.asarray(e.shape, dtype=float)
        if len(pts):
            corners.append(pts[0])
    if len(corners) < 2:
        return _piece_bbox_wh(piece)
    corners = np.asarray(corners)
    return (float(corners[:, 0].max() - corners[:, 0].min()),
            float(corners[:, 1].max() - corners[:, 1].min()))


def _grid_position_layout(pieces, cell, off, cons=None):
    """Fallback layout: place pieces on a regular grid. Used when the
    edge-translation approach collapses/overlaps pieces.

    Spacing uses each piece's CORE bbox (corner points only, see
    _piece_core_bbox_wh) rather than the full edge bbox, which is inflated by
    any protruding HEAD tab and produces visibly oversized gaps between
    pieces in the solved layout."""
    widths, heights = [], []
    for p in pieces:
        w, h = _piece_core_bbox_wh(p)
        widths.append(w)
        heights.append(h)
    piece_w = float(np.median(widths)) if widths else 100.0
    piece_h = float(np.median(heights)) if heights else 100.0

    rows = [cell[i][0] for i in range(len(pieces))]
    cols = [cell[i][1] for i in range(len(pieces))]
    r0, c0 = min(rows), min(cols)

    # Anchor: keep piece 0 at its current centroid.
    anchor_cents = []
    for p in pieces:
        xs, ys = [], []
        for e in p.edges_:
            arr = np.asarray(e.shape, dtype=float)
            if len(arr):
                xs.extend(arr[:, 0].tolist())
                ys.extend(arr[:, 1].tolist())
        anchor_cents.append(np.array([np.mean(xs) if xs else 0.0, np.mean(ys) if ys else 0.0]))

    anchor_xy = anchor_cents[0].copy()
    anchor_grid = (cell[0][0] - r0, cell[0][1] - c0)

    for i, p in enumerate(pieces):
        gr = cell[i][0] - r0
        gc = cell[i][1] - c0
        target_x = anchor_xy[0] + (gc - anchor_grid[1]) * piece_w
        target_y = anchor_xy[1] + (gr - anchor_grid[0]) * piece_h
        t = np.array([target_x - anchor_cents[i][0], target_y - anchor_cents[i][1]])

        for e in p.edges_:
            e.shape = np.round(np.asarray(e.shape, dtype=float) + t).astype(int)
        _translate_pixels_xy(p, t)


def _seam_quality_cost(pi, ki, pj, kj, green=False):
    """Score edge ki of pi against edge kj of pj the same way _precompute_seams
    does, for an edge pair that's only geometrically implied (faces the right
    grid direction) but was never actually scored/validated during the DFS.

    Why this is needed: a 6-piece (3x2) grid has 7 adjacent-cell pairs, but
    DFS only needs a 5-edge spanning tree of validated, low-cost seam matches
    to build that grid (legality is enforced structurally, not by re-scoring
    every cycle-closing adjacency). The remaining ~2 adjacencies fall out of
    the grid geometry "for free" — _geometric_layout's constraint loop will
    happily build a constraint from whatever edge happens to face the right
    direction, even if that specific pair was never checked for shape
    compatibility. If it's a poor match, its computed translation is
    essentially random, which can blow up the global least-squares solve."""
    pair = [pi, pj]
    snap = _snapshot(pair)
    ei, ej = pi.edges_[ki], pj.edges_[kj]
    stick_pieces(ei, pj, ej, centroid_bloc=_centroid(pi), centroid_cand=_centroid(pj))
    cost = _seam_cost(ej, ei, pj, pi, green)
    _restore(pair, snap)
    return cost


def _geometric_layout(pieces, cell, off, log=print, green=False):
    # Materialise the solved grid with stable placement.
    n = len(pieces)

    _axis_lock_all(pieces, off, log=log)

    # --- Post-hoc border-direction correction ---
    # After axis-lock, each piece's BORDER edges should physically point in the
    # expected exterior directions for their assigned grid cell.  A systematic
    # 90°/180° mismatch can occur when the DFS committed to an off value that
    # is structurally valid (passes _structure_ok) but geometrically wrong by
    # one step — typically because the seam-score difference between two
    # adjacent off values was smaller than the measurement noise on connector
    # shapes.  The BORDER edges are far more reliably classified than connector
    # shapes, so a disagreement between where the border edge IS pointing (post
    # axis-lock) and where it SHOULD point (from cell position) reveals the
    # error unambiguously, and the minimum-step correction (90, -90, or 180°)
    # is applied to the full piece geometry and tracked in small_solver_rotation_deg.
    _ph_rows = [cell[i][0] for i in range(n)]
    _ph_cols = [cell[i][1] for i in range(n)]
    _ph_r0, _ph_c0 = min(_ph_rows), min(_ph_cols)
    _ph_H = max(_ph_rows) - _ph_r0 + 1
    _ph_W = max(_ph_cols) - _ph_c0 + 1
    for i, p in enumerate(pieces):
        bi = _border_edge_info(p)          # geometric directions AFTER axis-lock
        if not bi:
            continue
        nr = cell[i][0] - _ph_r0
        nc = cell[i][1] - _ph_c0
        expected = _expected_borders_for_cell(nr, nc, _ph_H, _ph_W)
        if not expected:
            continue
        current = frozenset(geo for _, geo in bi)
        if current == expected:
            continue                        # already correct, nothing to fix
        c_ph = _centroid(p)
        for step_deg in (90.0, -90.0, 180.0):
            a = np.deg2rad(step_deg)
            R_ph = np.array([[np.cos(a), -np.sin(a)],
                             [np.sin(a),  np.cos(a)]], dtype=float)
            trial = frozenset(
                _snap_xy_to_cardinal(
                    (np.asarray(p.edges_[k].shape, dtype=float).mean(0) - c_ph) @ R_ph.T
                )
                for k, _ in bi
            )
            if trial == expected:
                log(f"[small/border-post-fix] piece{getattr(p,'id',i)}: "
                    f"geom={current} expected={expected} -> +{step_deg:.0f}° correction")
                for e in p.edges_:
                    e.shape = np.round(
                        (np.asarray(e.shape, dtype=float) - c_ph) @ R_ph.T + c_ph
                    ).astype(int)
                _apply_rigid_to_pixels(p, R_ph, c_ph)
                prev = float(getattr(p, "small_solver_rotation_deg", 0.0))
                p.small_solver_rotation_deg = _ssr_norm_deg(prev + step_deg)
                break
    # --- end post-hoc correction ---

    cellmap = {cell[i]: i for i in range(n)}
    cons = []
    # Generous: validated spanning-tree seams seen so far cost ~40-150;
    # 1000 only excludes pairs that were never really checked (the cycle-
    # closing edges) or are genuinely a bad match, not adds noise to good ones.
    SEAM_QUALITY_LIMIT = 1000.0

    for i in range(n):
        for k in range(4):
            d = _dir_of(k, off[i])
            j = cellmap.get((cell[i][0] + _STEP[d][0], cell[i][1] + _STEP[d][1]))
            if j is None or j <= i:
                continue

            k2 = next(kk for kk in range(4) if _dir_of(kk, off[j]) == _OPP[d])

            quality = _seam_quality_cost(pieces[i], k, pieces[j], k2, green)
            if not np.isfinite(quality) or quality > SEAM_QUALITY_LIMIT:
                log(f"[small/cons] SKIP piece{getattr(pieces[i],'id',i)}.e{k}({d}) <-> "
                    f"piece{getattr(pieces[j],'id',j)}.e{k2}({_OPP[d]}) "
                    f"quality={quality:.0f} (never validated by DFS / poor match)")
                continue

            delta = _edge_translation_delta(pieces[i].edges_[k], pieces[j].edges_[k2])
            cons.append((i, j, np.asarray(delta, dtype=float)))
            log(f"[small/cons] piece{getattr(pieces[i],'id',i)}.e{k}({d}) <-> "
                f"piece{getattr(pieces[j],'id',j)}.e{k2}({_OPP[d]}) "
                f"quality={quality:.0f} delta={tuple(np.round(delta,1))}")

    # Filtering out poor-quality seams can disconnect the constraint graph
    # (a piece with no remaining validated edge to any neighbour). The
    # least-squares solve below silently gives disconnected pieces an
    # arbitrary (effectively zero) translation, which would place them on
    # top of the anchor — so verify connectivity first and bail out to the
    # grid fallback if it's broken, rather than risk that arbitrarily.
    parent_cc = list(range(n))

    def _cc_find(x):
        while parent_cc[x] != x:
            parent_cc[x] = parent_cc[parent_cc[x]]
            x = parent_cc[x]
        return x

    for i, j, _ in cons:
        ri, rj = _cc_find(i), _cc_find(j)
        if ri != rj:
            parent_cc[ri] = rj

    connected = len({_cc_find(i) for i in range(n)}) == 1

    if not cons or not connected:
        if cons and not connected:
            log("[small/cons] filtered constraint graph is disconnected -> "
                "falling back to regular grid placement")
        _grid_position_layout(pieces, cell, off, cons=None)
        _ssg_global_assembly_snap(pieces)
        return

    sol = {}
    for axis in (0, 1):
        A = np.zeros((len(cons) + 1, n), dtype=float)
        rhs = np.zeros(len(cons) + 1, dtype=float)

        for r, (i, j, delta) in enumerate(cons):
            A[r, j] += 1.0
            A[r, i] -= 1.0
            rhs[r] = float(delta[axis])

        A[-1, 0] = 1.0
        rhs[-1] = 0.0
        sol[axis] = np.linalg.lstsq(A, rhs, rcond=None)[0]

    for i in range(n):
        t = np.array([sol[0][i], sol[1][i]], dtype=float)  # (x=col, y=row)
        for e in pieces[i].edges_:
            e.shape = np.round(np.asarray(e.shape, dtype=float) + t).astype(int)
        _translate_pixels_xy(pieces[i], t)

    # Überprüfe ob das Edge-Translation-Layout sinnvoll ist.
    #
    # Zwei unabhängige Gates, weil keiner allein zuverlässig genug ist:
    # 1) _layout_overlap_ratio: erkennt grobes Kollabieren/Kreuzen von Teilen
    #    (Pixel-basiert). Echte gelöste Puzzleteile überlappen praktisch nie
    #    messbar — ein einziges überlappendes Paar ist schon ein Warnsignal.
    # 2) _check_solution_is_rectangle: ein Spannbaum aus Kanten-Constraints
    #    hat per Konstruktion immer Residuum=0 (keine Zyklen, die eine falsche
    #    Verbindung aufdecken könnten) — selbst wenn eine einzelne Naht den
    #    FALSCHEN, aber ähnlich aussehenden Konnektor gematcht hat. Nur die
    #    GESAMTFORM (muss ein Rechteck sein) kann das noch auffangen.
    # Schlägt eines davon fehl, ist die gleichmässige Gitterplatzierung (mit
    # Kernmass ohne Konnektor-Nasen) die robustere Wahl.
    overlap_ratio = _layout_overlap_ratio(pieces)
    is_rect = _check_solution_is_rectangle(pieces, log=log)
    log(f"[small/layout] overlap_ratio={overlap_ratio:.2f} is_rectangle={is_rect} "
        f"translations={[(round(float(sol[0][i]), 1), round(float(sol[1][i]), 1)) for i in range(n)]}")
    if overlap_ratio > 0.05 or not is_rect:
        log("[small/layout] edge-translation result rejected -> falling back to regular grid placement")
        # Undo translations: restore to post-axislock positions via grid fallback.
        for i in range(n):
            t = np.array([-sol[0][i], -sol[1][i]], dtype=float)
            for e in pieces[i].edges_:
                e.shape = np.round(np.asarray(e.shape, dtype=float) + t).astype(int)
            _translate_pixels_xy(pieces[i], t)
        _grid_position_layout(pieces, cell, off, cons=cons)

    _ssg_global_assembly_snap(pieces)
def _ssg_collect_piece_points_xy(pieces):
    # Collect all visible piece points as (x=col, y=row).
    pts_all = []

    for p in pieces:
        if getattr(p, "pixels", None):
            for row, col in p.pixels.keys():
                pts_all.append([float(col), float(row)])
        else:
            for e in getattr(p, "edges_", []):
                arr = np.asarray(e.shape, dtype=float)
                if len(arr):
                    pts_all.extend(arr.tolist())

    if not pts_all:
        return np.zeros((0, 2), dtype=float)

    return np.asarray(pts_all, dtype=float)


def _ssg_rotate_pixels(piece, R, center_xy):
    if not getattr(piece, "pixels", None):
        return

    c = np.asarray(center_xy, dtype=float)
    new_pixels = {}

    for (row, col), color in piece.pixels.items():
        p_xy = np.array([float(col), float(row)], dtype=float)
        q_xy = (p_xy - c) @ R.T + c
        new_col = int(round(q_xy[0]))
        new_row = int(round(q_xy[1]))
        new_pixels[(new_row, new_col)] = color

    piece.pixels = new_pixels


def _ssg_rotate_edges(piece, R, center_xy):
    c = np.asarray(center_xy, dtype=float)

    for e in getattr(piece, "edges_", []):
        arr = np.asarray(e.shape, dtype=float)
        if len(arr):
            e.shape = np.round((arr - c) @ R.T + c).astype(int)


def _ssg_global_assembly_snap(pieces):
    # Rotate the complete solved assembly back to a horizontal A5 layout.
    #
    # Wichtig:
    # Diese Funktion dreht alle Teile gemeinsam um denselben Mittelpunkt.
    # Dadurch bleibt die relative Puzzle-Lösung erhalten.
    pts = _ssg_collect_piece_points_xy(pieces)
    if len(pts) < 20:
        return

    center = pts.mean(axis=0)
    X = pts - center

    # PCA: lange Hauptachse des kompletten gelösten Puzzles bestimmen.
    cov = X.T @ X
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]

    angle = float(np.degrees(np.arctan2(v[1], v[0])))

    # Winkel der Hauptachse auf [-90, 90] normalisieren.
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0

    # Für das Wettbewerbspuzzle soll die lange Achse horizontal liegen.
    # Deshalb komplette Assembly um -angle drehen.
    delta = -angle

    # Schutz: Wenn PCA etwas völlig Unplausibles liefert, nicht eingreifen.
    if abs(delta) < 0.5:
        return
    if abs(delta) > 45.0:
        return

    a = np.deg2rad(delta)
    R = np.array([
        [np.cos(a), -np.sin(a)],
        [np.sin(a),  np.cos(a)],
    ], dtype=float)

    for p in pieces:
        _ssg_rotate_edges(p, R, center)
        _ssg_rotate_pixels(p, R, center)

        # Auch diese globale Drehung gehört zur späteren Roboterrotation.
        prev_solver_rot = float(getattr(p, "small_solver_rotation_deg", 0.0))
        p.small_solver_rotation_deg = _ssr_norm_deg(prev_solver_rot + float(delta))



# --------------------------------------------------------------------------- #
# Border-based grid assignment — the primary strategy for any rectangular puzzle.
#
# Key insight: flat (BORDER) edges of a puzzle piece are far more reliably
# detected than connector (HEAD/HOLE) shapes. For any rectangular grid, the
# DIRECTION of a piece's border edges relative to its centroid directly encodes
# which grid position and orientation the piece needs — without any shape matching.
#
# Example for a 2×3 grid:
#   border pointing N+W  → top-left corner  (0,0)
#   border pointing N    → top edge         (0,1)
#   border pointing N+E  → top-right corner (0,2)
#   border pointing S+W  → bottom-left      (1,0)
#   border pointing S    → bottom edge      (1,1)
#   border pointing S+E  → bottom-right     (1,2)
#
# This works for any grid size. The DFS / seam-scoring is used only as fallback
# when border directions are ambiguous or edge classification is unreliable.
# --------------------------------------------------------------------------- #

def _snap_xy_to_cardinal(vec_xy):
    """Snap a 2D (x=col, y=row) vector to the nearest cardinal direction string."""
    x, y = float(vec_xy[0]), float(vec_xy[1])
    if abs(x) >= abs(y):
        return 'E' if x >= 0 else 'W'
    return 'S' if y >= 0 else 'N'  # y positive = down = South in image coords


def _border_edge_info(piece):
    """Return list of (edge_index, geometric_cardinal_dir) for all BORDER edges."""
    c = _centroid(piece)
    result = []
    for k, e in enumerate(piece.edges_):
        if e.type != TypeEdge.BORDER:
            continue
        pts = np.asarray(e.shape, dtype=float)
        if len(pts) == 0:
            continue
        vec = pts.mean(axis=0) - c
        if np.linalg.norm(vec) < 1e-6:
            continue
        result.append((k, _snap_xy_to_cardinal(vec)))
    return result


def _expected_borders_for_cell(row, col, H, W):
    """Frozenset of expected border directions for grid cell (row, col) in H×W grid."""
    return frozenset(
        d for d, cond in [('N', row == 0), ('S', row == H - 1),
                           ('W', col == 0), ('E', col == W - 1)]
        if cond
    )


def _border_dirs_given_off(border_info, off):
    """Which cardinal directions would the border edges face at this orientation offset?"""
    return frozenset(_dir_of(k, off) for k, _ in border_info)


def _assign_grid_from_borders(pieces, cands, log=print):
    """Assign grid positions and orientation offsets from border-edge geometry.

    For each piece we try all 4 orientation offsets (0..3).  For each offset we
    compute which cardinal directions the border edges would face and look for an
    unoccupied grid cell with a matching expected-border set.

    Pieces that have multiple possible cells (due to identical border patterns,
    e.g. two edge pieces that both could face N) are handled by trying all orderings
    — the algorithm backtracks at depth-first rather than greedily committing.

    Returns (cell_dict, off_dict) on success, or None.
    """
    n = len(pieces)
    border_info = [_border_edge_info(p) for p in pieces]
    for i in range(n):
        log(f"[small/border-raw] piece {getattr(pieces[i], 'id', i)}: "
            f"raw_border_edges={border_info[i]} nBorders_={getattr(pieces[i], 'nBorders_', '?')}")

    for W, H in cands:
        result = _dfs_border_assign(border_info, W, H, n)
        if result is not None:
            cell_res, off_res = result
            log(f"[small/border] {n} pieces -> {W}x{H} grid via border directions")
            for i in range(n):
                log(f"[small/border]   piece {getattr(pieces[i], 'id', i)}: "
                    f"cell={cell_res[i]}, off={off_res[i]}, "
                    f"borders={set(_border_dirs_given_off(border_info[i], off_res[i]))}")
            return cell_res, off_res

    return None


def _dfs_border_assign(border_info, W, H, n):
    """DFS over piece ordering to assign cells + offsets from border directions.

    Backtracking is needed for ambiguous cases (two edge pieces with the same
    expected border, e.g. when the grid is 3×2 and two pieces both face 'W').
    """
    expected = {(r, c): _expected_borders_for_cell(r, c, H, W)
                for r in range(H) for c in range(W)}

    cell = {}
    off_dict = {}

    def recurse(i, used_cells):
        if i == n:
            return True
        bi = border_info[i]
        for try_off in range(4):
            dirs = _border_dirs_given_off(bi, try_off)
            for (r, c), exp in expected.items():
                if exp == dirs and (r, c) not in used_cells:
                    cell[i] = (r, c)
                    off_dict[i] = try_off
                    used_cells.add((r, c))
                    if recurse(i + 1, used_cells):
                        return True
                    used_cells.discard((r, c))
        return False

    if recurse(0, set()):
        return dict(cell), dict(off_dict)
    return None


def solve_small(pieces, green=False, log=print):
    """Assemble a small (4-9 piece, w×h ≥2 grid) puzzle.

    PRIMARY PATH — seam-scoring DFS over connector (HEAD/HOLE) shape matches:
      This is the only signal that can determine WHICH piece is adjacent to
      WHICH other piece. Border (flat-edge) direction alone cannot do this:
      a corner piece's two flat edges look identical regardless of which of
      the rectangle's 4 corners it actually belongs to (rotating the piece
      makes its border pair face any corner) — same for edge pieces and the
      rectangle's edge cells. Border direction only fixes the ORIENTATION
      (off) once a cell is otherwise known, and the grid-merge legality
      check (no BORDER edge may face an occupied/interior cell) already
      enforces "corners only in corner cells" as a hard constraint during
      the DFS — so connector matching plus that constraint is both necessary
      and sufficient; border direction can't replace it.

    Mutates piece shapes into the solved layout and sets p.coord=(row,col).
    """
    n = len(pieces)
    nC = sum(1 for p in pieces if p.nBorders_ == 2)
    nE = sum(1 for p in pieces if p.nBorders_ == 1)
    nX = sum(1 for p in pieces if p.nBorders_ == 0)
    cands = _candidate_dims(n, nC, nE, nX) if 4 <= n <= 9 else []
    if not cands:
        nOver = sum(1 for p in pieces if p.nBorders_ > 2)
        log(f"[small] not applicable: {n} pieces, {nC} corners / {nE} edges / {nX} centers "
            f"— no rectangle matches this classification")
        log(f"[small] per-piece border counts: {[p.nBorders_ for p in pieces]}")
        if nOver:
            log(f"[small] {nOver} piece(s) have >2 flat edges — upstream edge classification "
                f"over-flagged connectors as BORDER (a real piece has at most 2 flat sides).")
        return False
    log(f"[small] candidate grid dim(s): {cands}")

    orig = _snapshot(pieces)

    # Diagnostic only: log what border directions alone would suggest (useful
    # to confirm corner/edge type counts are sane), but never use this as the
    # cell assignment — see docstring for why that would be wrong.
    _assign_grid_from_borders(pieces, cands, log=log)

    parent = list(range(n))
    members = {i: [i] for i in range(n)}
    cell_tmp = {i: (0, 0) for i in range(n)}
    off_tmp = {i: 0 for i in range(n)}

    seams = _precompute_seams(pieces, green)
    log("[small] {} candidate seams; best costs: {}".format(
        len(seams), ", ".join(f"{s['cost']:.0f}" for s in seams[:8])))

    best = {'cost': float('inf'), 'cell': None, 'off': None}
    _dfs(seams, 0, pieces, parent, members, cell_tmp, off_tmp,
         0.0, [300000], best, cands)

    # Second pass with shuffled near-tied seams for diversity (helps when the
    # best-first order commits early to a locally-cheap but globally wrong seam).
    if n >= 6 and len(seams) >= 4:
        cost_threshold = seams[0]['cost'] * 1.20 if seams else 0.0
        seams2 = [s for s in seams if s['cost'] <= cost_threshold]
        rest = [s for s in seams if s['cost'] > cost_threshold]
        import random as _rng
        rng = _rng.Random(42)
        rng.shuffle(seams2)
        seams_shuffled = seams2 + rest
        parent2 = list(range(n)); members2 = {i: [i] for i in range(n)}
        cell2 = {i: (0, 0) for i in range(n)}; off2 = {i: 0 for i in range(n)}
        _dfs(seams_shuffled, 0, pieces, parent2, members2, cell2, off2,
             0.0, [300000], best, cands)

    if best['cell'] is None:
        _restore(pieces, orig)
        log("[small] no valid grid assembly found")
        return False

    cell = best['cell']
    off_dict = best['off']
    log(f"[small] DFS best assembly cost = {best['cost']:.0f}")
    for i in range(n):
        log(f"[small/dfs-place] piece{getattr(pieces[i], 'id', i)}: "
            f"cell={cell[i]}, off={off_dict[i]}")

    rows = [cell[i][0] for i in range(n)]
    cols = [cell[i][1] for i in range(n)]
    r0, c0 = min(rows), min(cols)
    for i, p in enumerate(pieces):
        p.coord = (cell[i][0] - r0, cell[i][1] - c0)

    _geometric_layout(pieces, cell, off_dict, log=log)

    if not _check_solution_is_rectangle(pieces, log=log):
        log("[small] WARNING: solved assembly failed the rectangle check — "
            "grid topology may be right but piece rotation/placement is off")

    log("[small] solved (agglomerative + grid embedding)")
    return True