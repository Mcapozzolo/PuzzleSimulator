import itertools
import math
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

from solver.Puzzle.Edge import Edge
from solver.Puzzle.Enums import directions, TypeEdge
from solver.Puzzle.PuzzlePiece import PuzzlePiece
from .peak_detect import detect_peaks
from solver.Puzzle.Distance import rgb2hsl
from multiprocessing import Pool, cpu_count
import itertools
from .. import config


COUNT = 0


def get_relative_angles(cnt, export=False, sigma=5):
    """
    Berechnet die relativen Winkeländerungen entlang einer Kontur.

    - cnt: Liste der (x,y)-Punkte der Kontur
    - sigma: Glättungsparameter des Gaußfilters (höher = stärker geglättet)
    - export: Wenn True → speichert die Signatur und Diagramm zur Kontrolle

    Rückgabe:
    - Array von Winkeländerungen (relative Winkelkurve)
    """

    global COUNT
    COUNT = COUNT + 1

    length = len(cnt)
    angles = []
    last = np.pi

    cnt_tmp = np.array(cnt)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    for i in range(0, len(cnt) - 1):
        dir = (cnt[i + 1][0] - cnt[i][0], cnt[i + 1][1] - cnt[i][1])
        angle = math.atan2(-dir[1], dir[0])
        while angle < last - np.pi:
            angle += 2 * np.pi
        while angle > last + np.pi:
            angle -= 2 * np.pi
        angles.append(angle)
        last = angle

    angles = np.diff(angles)

    k = [0.33, 0.33, 0.33, 0.33, 0.33]
    angles = scipy.ndimage.convolve(angles, k, mode="constant", cval=0.0)
    angles = scipy.ndimage.filters.gaussian_filter(angles, sigma)

    angles = np.roll(np.array(angles), -length)
    angles = angles[0:length]

    if export:
        pickle.dump(
            angles,
            open(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "save" + str(COUNT) + ".p"),
                "wb",
            ),
        )
        plt.plot(np.append(angles, angles))
        plt.savefig(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "fig" + str(COUNT) + ".png")
        )
        plt.clf()
        plt.cla()
        plt.close()

    return angles


def _classify_edge_by_convexity(edge_pts_xy, piece_centroid_xy):
    """Fallback HEAD/HOLE classification by raw geometry, used when the
    curvature-peak counting in type_peak() can't find a clean 2-peak pattern
    (e.g. due to contour noise). Finds the point of max perpendicular
    deviation from the edge's chord (the connector apex) and checks whether
    it bulges away from the piece centroid (HEAD, a protruding tab) or
    toward it (HOLE, an indentation) — this only needs the piece's own
    centroid, so it is robust where cross-piece comparisons are not."""
    pts = np.asarray(edge_pts_xy, dtype=np.float64)
    if len(pts) < 3:
        return TypeEdge.UNDEFINED
    a, b = pts[0], pts[-1]
    chord = b - a
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return TypeEdge.UNDEFINED
    u = chord / length
    normal = np.array([-u[1], u[0]])
    devs = (pts - a) @ normal
    idx = int(np.argmax(np.abs(devs)))
    apex_dev = float(devs[idx])
    if abs(apex_dev) < 1.5:
        return TypeEdge.BORDER
    apex = pts[idx]
    centroid = np.asarray(piece_centroid_xy, dtype=np.float64)
    to_centroid = centroid - a
    centroid_side = float(np.dot(to_centroid, normal))
    # Apex bulging on the SAME side as the centroid = indentation = HOLE.
    # Apex bulging on the OPPOSITE side = protruding tab = HEAD.
    if apex_dev * centroid_side > 0:
        return TypeEdge.HOLE
    return TypeEdge.HEAD


def _rectangle_score(pts):
    """Sum of squared cosines of interior angles (0 = perfect rectangle).

    pts: (4, 2) float array of corner coordinates in contour traversal order.
    A perfect rectangle has cos(90°)=0 at every corner, so the total is 0.

    This is a far more robust corner-validity check than counting head/hole
    peaks between candidate corners (the old approach): it directly measures
    "do these 4 points actually form a rectangle", independent of how the
    connector tabs/notches between them happen to be shaped, so it can't be
    fooled by a connector whose peak count doesn't match the expected 0/2/3
    pattern (the root cause of the old TypeEdge.UNDEFINED failures).
    """
    total = 0.0
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return float('inf')
        cos_a = np.dot(v1, v2) / (n1 * n2)
        total += cos_a * cos_a
    return total


def _find_corners_curvature_peaks(cnt, green=False):
    """Find the 4 piece corners as the top positive curvature peaks.

    Piece corners are sharp ~90° left-turns in the contour, producing the
    largest positive peaks in the smoothed angular-velocity signal. Head and
    hole shoulder peaks are smaller and wash out at higher sigma values.

    At each sigma we evaluate all C(K,4) spacing-valid combinations and score
    each by how closely the 4 points form a rectangle (_rectangle_score). The
    global best-scoring combination across all sigma levels is returned, with
    early exit once the rectangle score is below a tight threshold.

    Returns a sorted numpy int array of 4 contour indices, or None on failure.
    """
    n = len(cnt)
    cnt_pts = cnt[:, 0, :].astype(float)
    cnt_convert = [c[0] for c in cnt]
    OFFSET_LOW = n / 10
    OFFSET_HIGH = n / 2.0
    RECT_THRESHOLD = 0.05   # ~cos²(13°) per corner on average; exit early if beaten

    sigma = 5
    max_sigma = 12 if green else 20
    best_corners = None
    best_score = float('inf')

    while sigma <= max_sigma:
        ra = np.array(get_relative_angles(np.array(cnt_convert), sigma=sigma))
        if np.max(ra) <= 0:
            sigma += 1
            continue

        # Collect positive peaks with double-roll trick
        extr = detect_peaks(ra, mph=0.3 * np.max(ra))
        ra_sh = np.roll(ra, int(n / 2))
        extr = np.unique(np.append(
            extr,
            (detect_peaks(ra_sh, mph=0.3 * float(max(ra_sh))) - int(n / 2)) % n,
        ))

        if len(extr) < 4:
            sigma += 1
            continue

        # Limit to top-K by amplitude; C(K,4) << P(extr,4)
        K = min(len(extr), 8)
        top_k = np.sort(extr[np.argsort(ra[extr])[::-1]][:K])

        for comb in itertools.combinations(top_k, 4):
            c = np.array(comb)
            spacings = np.array([(c[(i + 1) % 4] - c[i]) % n for i in range(4)])
            if not (np.all(spacings > OFFSET_LOW) and np.all(spacings < OFFSET_HIGH)):
                continue
            score = _rectangle_score(cnt_pts[c])
            if score < best_score:
                best_score = score
                best_corners = c

        if best_score < RECT_THRESHOLD:
            break   # already a tight rectangle, no need to keep smoothing

        sigma += 1

    return best_corners


def _classify_edges_by_baseline(edges, center, flat_frac=None):
    """HEAD/HOLE/BORDER per edge from deviation of the curve vs its
    corner-to-corner baseline, signed against the outward (away-from-centre) dir.

    edges:  list of 4 (M,2) arrays of (col,row) contour points, each starting and
            ending at a piece corner.
    center: (cx, cy) piece centre in the same coordinate space.
    flat_frac: max |apex deviation| / baseline length below which the edge is flat.

    This sidesteps the old peak-counting classification entirely: instead of
    requiring a specific count of curvature peaks between corners (fragile when
    contour noise adds/removes a peak), it measures the actual signed shape
    deviation of the whole edge from a straight line, which directly and
    unambiguously distinguishes flat/bulging-out/bulging-in.
    """
    if flat_frac is None:
        flat_frac = getattr(config, "EDGE_FLAT_FRAC", 0.13)
    center = np.asarray(center, dtype=float)
    types = []
    for pts in edges:
        pts = np.asarray(pts, dtype=float)
        a, b = pts[0], pts[-1]                       # the two corners
        base = b - a
        base_len = np.linalg.norm(base)
        if base_len < 1e-6 or len(pts) < 3:
            types.append(TypeEdge.UNDEFINED)
            continue
        n_hat = np.array([-base[1], base[0]]) / base_len   # unit normal to baseline
        signed = (pts - a) @ n_hat                    # signed perp. offset per point
        m = max(1, int(0.10 * len(pts)))              # ignore corner-rounding margin
        interior = signed[m:-m] if len(signed) > 2 * m else signed
        k = int(np.argmax(np.abs(interior)))
        max_dev = interior[k]                          # apex deviation (signed)
        if abs(max_dev) / base_len < flat_frac:
            types.append(TypeEdge.BORDER)
            continue
        mid = (a + b) / 2.0
        outward = mid - center                        # away-from-centre direction
        apex_disp = n_hat * max_dev                   # actual deviation vector
        types.append(TypeEdge.HEAD if np.dot(apex_disp, outward) > 0
                     else TypeEdge.HOLE)
    return types


def _nearest_contour_index(point_xy, contour_xy):
    px, py = point_xy
    d2 = np.sum((contour_xy - np.array([px, py])) ** 2, axis=1)
    return int(np.argmin(d2))


def _build_edges_from_corner_indices(cnt_arr, corner_idx):
    corner_idx = np.array(sorted(set(int(i) for i in corner_idx)), dtype=int)
    if len(corner_idx) != 4:
        return None

    edges_local = []
    for i in range(3):
        a = corner_idx[i]
        b = corner_idx[i + 1]
        if a >= b:
            return None
        edges_local.append(cnt_arr[a:b])

    a = corner_idx[3]
    b = corner_idx[0]
    edges_local.append(np.concatenate((cnt_arr[a:], cnt_arr[:b]), axis=0))

    edges_local = [np.array([x[0] for x in e]) for e in edges_local]
    return edges_local


def _classify_fallback_edges(edges_local):
    """Classify fallback edges by straightness.
    Very straight edges are BORDER, curved/non-straight edges are
    classified by apex-convexity (instead of a blanket UNDEFINED)."""
    types = []

    deviations = []
    for edge in edges_local:
        pts = np.asarray(edge, dtype=np.float32)
        if len(pts) < 2:
            deviations.append(float("inf"))
            continue

        a = pts[0]
        b = pts[-1]
        ab = b - a
        norm = np.linalg.norm(ab)

        if norm < 1e-6:
            deviations.append(float("inf"))
            continue

        # distance of all points to line a-b
        ap = pts - a
        dist = np.abs(np.cross(ab, ap) / norm)
        deviations.append(float(np.mean(dist)))

    # Smaller = straighter
    sorted_dev = sorted(deviations)

    # For your puzzle pieces, usually 2 sides are outer straight borders.
    border_threshold = max(3.5, sorted_dev[1] * 1.35) if len(sorted_dev) >= 2 else 3.5

    centroid = np.mean(
        np.concatenate([np.asarray(e, dtype=np.float64) for e in edges_local], axis=0),
        axis=0,
    )
    for dev, edge in zip(deviations, edges_local):
        if dev <= border_threshold:
            types.append(TypeEdge.BORDER)
        else:
            types.append(_classify_edge_by_convexity(edge, centroid))

    return types


def _fallback_signature_from_polygon(cnt_local):
    """Geometric fallback when curvature-peak corner detection fails."""
    peri = cv2.arcLength(cnt_local, True)

    approx = None
    for eps_factor in [0.01, 0.015, 0.02, 0.03, 0.04]:
        cand = cv2.approxPolyDP(cnt_local, eps_factor * peri, True)
        if len(cand) >= 4:
            approx = cand
            break

    if approx is None:
        approx = cnt_local

    pts = np.array([p[0] for p in approx], dtype=np.int32)

    if len(pts) > 4:
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]

        candidates = [
            pts[np.argmin(s)],  # tl
            pts[np.argmax(d)],  # tr
            pts[np.argmax(s)],  # br
            pts[np.argmin(d)],  # bl
        ]

        uniq = []
        seen = set()
        for p in candidates:
            key = (int(p[0]), int(p[1]))
            if key not in seen:
                uniq.append(np.array(key))
                seen.add(key)

        pts = np.array(uniq, dtype=np.int32)

    contour_xy = np.array([p[0] for p in cnt_local], dtype=np.int32)

    if len(pts) < 4:
        x, y, w, h = cv2.boundingRect(cnt_local)
        pts = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.int32,
        )

    corner_idx = [_nearest_contour_index(p, contour_xy) for p in pts]
    corner_idx = sorted(set(corner_idx))

    if len(corner_idx) < 4:
        n = len(cnt_local)
        corner_idx = [0, n // 4, n // 2, (3 * n) // 4]

    if len(corner_idx) > 4:
        idx = np.array(corner_idx, dtype=int)
        selected = [idx[0]]
        while len(selected) < 4:
            remaining = [x for x in idx if x not in selected]
            best = None
            best_score = -1
            for r in remaining:
                score = min(abs(r - s) for s in selected)
                if score > best_score:
                    best_score = score
                    best = r
            selected.append(best)
        corner_idx = sorted(selected)

    edges_local = _build_edges_from_corner_indices(cnt_local, corner_idx)
    if edges_local is None:
        return None, None, None

    types_local = _classify_fallback_edges(edges_local)

    return np.array(corner_idx, dtype=int), edges_local, types_local


def my_find_corner_signature(cnt, green=False):
    """Determine corner/edge positions by analyzing a piece contour.

    Corners are found via curvature-peak + rectangle-score search
    (_find_corners_curvature_peaks), and edges are classified by their signed
    deviation from the corner-to-corner baseline (_classify_edges_by_baseline).
    This replaced the older peak-counting approach (type_peak/is_acceptable_comb),
    which produced TypeEdge.UNDEFINED whenever a connector's curvature peaks didn't
    match an expected count — a frequent failure on real (noisy) contours.
    """
    corner_indices = _find_corners_curvature_peaks(cnt, green)
    if corner_indices is None:
        print(f"[corner-detect] curvature peak search FAILED (contour len={len(cnt)})")
        return _fallback_signature_from_polygon(cnt)

    n = len(cnt)
    offset = n - int(corner_indices[3]) - 1
    best_fit = corner_indices + offset
    best_fit_tmp = corner_indices

    edges = []
    for i in range(3):
        edges.append(cnt[best_fit_tmp[i]:best_fit_tmp[i + 1]])
    edges.append(np.concatenate((cnt[best_fit_tmp[3]:], cnt[:best_fit_tmp[0]]), axis=0))
    edges = [np.array([x[0] for x in e]) for e in edges]

    M = cv2.moments(cnt)
    center = (M['m10'] / M['m00'], M['m01'] / M['m00']) if M['m00'] != 0 \
        else tuple(np.mean(np.concatenate(edges, axis=0), axis=0))
    final_types = _classify_edges_by_baseline(edges, center)

    if TypeEdge.UNDEFINED in final_types:
        piece_centroid = np.mean(np.array([p[0] for p in cnt], dtype=np.float64), axis=0)
        fixed_types = []
        for t, e in zip(final_types, edges):
            if t == TypeEdge.UNDEFINED:
                t = _classify_edge_by_convexity(e, piece_centroid)
                print(f"[Extractor] UNDEFINED edge reclassified by convexity fallback -> {t}")
            fixed_types.append(t)
        final_types = fixed_types

    return best_fit, edges, final_types


def export_contours_without_colormatching(
    img, img_bw, contours, modulo, export_img=True
):
    puzzle_pieces = []
    list_img = []

    # Windows-safe: no multiprocessing
    signatures = []
    for cnt in contours:
        try:
            signatures.append(my_find_corner_signature(cnt, False))
        except Exception as e:
            print(f"[Extractor] corner signature failed for one contour: {e}")
            signatures.append((None, None, None))

    for idx, cnt in enumerate(contours):
        corners, edges_shape, types_edges = signatures[idx]

        if corners is None or edges_shape is None or types_edges is None:
            print(f"[Extractor] contour {idx}: no usable signature")
            continue

        if len(edges_shape) != 4 or len(types_edges) != 4:
            print(f"[Extractor] contour {idx}: invalid edge package")
            continue

        mask_border = np.zeros_like(img_bw)
        mask_full = np.zeros_like(img_bw)

        mask_full = cv2.drawContours(mask_full, contours, idx, 255, -1)
        mask_border = cv2.drawContours(mask_border, contours, idx, 255, 1)

        img_piece = np.zeros_like(img)
        img_piece[mask_full == 255] = img[mask_full == 255]

        xs, ys = np.where(mask_full == 255)
        pixels = {(x, y): img_piece[x, y] for x, y in zip(xs, ys)}

        try:
            edges = [
                Edge(
                    s,
                    None,
                    edge_type=types_edges[i],
                    direction=directions[i],
                    connected=types_edges[i] == TypeEdge.BORDER,
                )
                for i, s in enumerate(edges_shape)
            ]

            puzzle_pieces.append(PuzzlePiece(edges, pixels))
        except Exception as e:
            print(f"[Extractor] contour {idx}: PuzzlePiece creation failed: {e}")
            continue

        mask_border = np.zeros_like(img_bw)
        for i in range(4):
            for p in edges_shape[i]:
                py, px = int(p[1]), int(p[0])
                if 0 <= py < mask_border.shape[0] and 0 <= px < mask_border.shape[1]:
                    mask_border[py, px] = 255

        out = np.zeros_like(img_bw)
        out[mask_border == 255] = img_bw[mask_border == 255]

        x, y, w, h = cv2.boundingRect(cnt)
        out2 = out[y : y + h, x : x + w]
        list_img.append(out2)

    if not list_img:
        return puzzle_pieces, None

    max_height = max(x.shape[0] for x in list_img)
    max_width = max(x.shape[1] for x in list_img)

    pieces_img = np.zeros(
        [max_height * (int(len(list_img) / modulo) + 1), max_width * modulo],
        dtype=np.uint8,
    )

    for index, image in enumerate(list_img):
        pieces_img[
            (max_height * int(index / modulo)) : (
                max_height * int(index / modulo) + image.shape[0]
            ),
            (max_width * (index % modulo)) : (
                max_width * (index % modulo) + image.shape[1]
            ),
        ] = image

    return puzzle_pieces, pieces_img
