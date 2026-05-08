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


def is_maximum_local(index, relative_angles, radius):
    """
    Determine if a point at index is a maximum local in radius range of relative_angles function

    :param index: index of the point to check in relative_angles list
    :param relative_angles: list of angles
    :param radius: radius used to check neighbors
    :return: Boolean
    """

    start = max(0, index - radius)
    end = min(relative_angles.shape[0] - 1, index + radius)
    for i in range(start, end + 1):
        if relative_angles[i] > relative_angles[index]:
            return False
    return True


def longest_peak(relative_angles):
    """
    Find the longest area < 0

    :param relative_angles: list of angles
    :return: coordinates of the area
    """

    length = relative_angles.shape[0]
    longest = (0, 0)
    j = 0
    for i in range(length):
        if relative_angles[i] >= 0:
            j = i
        if i - j > longest[1] - longest[0]:
            longest = (j, i)
    return longest


def distance_signature(relative_angles):
    """
    Distance of each points to the line formed by first and last points

    :param relative_angles: list of angles
    :return: List of floats
    """
    flat_angles = relative_angles.flatten()
    length = flat_angles.shape[0]

    l1 = np.array([0, flat_angles[0]])
    l2 = np.array([length - 1, flat_angles[-1]])
    assert np.linalg.norm(l2 - l1) != 0

    signature = np.zeros((length, 1))

    for i in range(length):
        signature[i] = np.linalg.norm(
            np.cross(l2 - l1, l1 - np.array([i, flat_angles[i]]))
        ) / np.linalg.norm(l2 - l1)

    return signature


def flat_score(relative_angles):
    """
    Compute the flat score of relative_angles

    :param relative_angles: list of angles
    :return: List of floats
    """

    length = relative_angles.shape[0]
    distances = distance_signature(relative_angles)
    diff = 0
    for i in range(length):
        diff = max(diff, abs(distances[i]))
    return diff


def indent_score(relative_angles):
    """
    Compute score for indent part

    :param relative_angles: list of angles
    :return: List of floats
    """

    length = relative_angles.shape[0]
    peak = longest_peak(relative_angles)

    while peak[0] > 0 and not is_maximum_local(peak[0], relative_angles, 10):
        peak = (peak[0] - 1, peak[1])
    while peak[1] < length - 1 and not is_maximum_local(peak[1], relative_angles, 10):
        peak = (peak[0], peak[1] + 1)

    shape = np.zeros((peak[0] + length - peak[1], 1))
    for i in range(peak[0] + 1):
        shape[i] = relative_angles[i]
    for i in range(peak[1], length):
        shape[i - peak[1] + peak[0]] = relative_angles[i]

    # FIX FOR FUNCTIONS > 0
    if shape.shape[0] == 1:
        return flat_score(relative_angles)
    return flat_score(shape)


def outdent_score(relative_angles):
    """
    Compute score for outdent part

    :param relative_angles: list of angles
    :return: List of floats
    """
    return indent_score(-relative_angles)


def compute_comp(combs_l, relative_angles, method="correlate"):
    """
    Compute score for each combination of 4 points and return the index of the best

    :param combs_l: list of combinations of 4 points
    :param relative_angles: List of angles
    :return: Int
    """

    results_glob = []
    for comb_t in combs_l:
        # Roll the values of relative angles for this combination
        offset = len(relative_angles) - comb_t[3] - 1
        relative_angles_tmp = np.roll(relative_angles, offset)
        comb_t += offset
        comb_t = [
            (0, comb_t[0]),
            (comb_t[0], comb_t[1]),
            (comb_t[1], comb_t[2]),
            (comb_t[2], comb_t[3]),
        ]

        results_comp = []
        for comb in comb_t:
            hole, head, border = 0, 0, 0
            if method == "flat":
                hole = indent_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
                head = outdent_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
                border = flat_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
            if hole != border:
                results_comp.append(min(hole, head))
            else:
                results_comp.append(border)
        results_glob.append(np.sum(np.array(results_comp)))
    return np.argmin(np.array(results_glob))


def peaks_inside(comb, peaks):
    """
    Check the number of peaks inside comb

    :param comb: Tuple of coordinates
    :param peaks: List of peaks to check
    :return: Int
    """
    if len(comb) == 0:
        return []
    return [peak for peak in peaks if peak > comb[0] and peak < comb[-1]]


def is_pattern(comb, peaks):
    """
    Check if the peaks formed an outdent or an indent pattern

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :return: Int
    """
    cpt = len(peaks_inside(comb, peaks))
    return cpt == 0 or cpt == 2 or cpt == 3


def is_acceptable_comb(combs, peaks, length):
    """
    Check if a combination is composed of acceptable patterns.
    Used to filter the obviously bad combinations quickly.

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :param length: Length of the signature (used for offset computation)
    :return: Boolean
    """

    offset = length - combs[3] - 1
    combs_tmp = combs + offset
    peaks_tmp = (peaks + offset) % length
    return (
        is_pattern([0, combs_tmp[0]], peaks_tmp)
        and is_pattern([combs_tmp[0], combs_tmp[1]], peaks_tmp)
        and is_pattern([combs_tmp[1], combs_tmp[2]], peaks_tmp)
        and is_pattern([combs_tmp[2], combs_tmp[3]], peaks_tmp)
    )


def type_peak(peaks_pos_inside, peaks_neg_inside):
    """
    Determine the type of lists of pos and neg peaks

    :param peaks_pos_inside: List of positive peaks
    :param peaks_neg_inside: List of negative peaks
    :return: TypeEdge
    """

    if len(peaks_pos_inside) == 0 and len(peaks_neg_inside) == 0:
        return TypeEdge.BORDER
    if len(peaks_inside(peaks_pos_inside, peaks_neg_inside)) == 2:
        return TypeEdge.HOLE
    if len(peaks_inside(peaks_neg_inside, peaks_pos_inside)) == 2:
        return TypeEdge.HEAD
    return TypeEdge.UNDEFINED


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, ord=order, axis=axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def my_find_corner_signature(cnt, green=False):
    """
    Determine the corner/edge positions by analyzing contours.

    Returns:
        corners_idx, edges, types_pieces
    """

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
        """
        Classify fallback edges by straightness.
        Very straight edges are BORDER, curved/non-straight edges are UNDEFINED.
        """
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

        for dev in deviations:
            if dev <= border_threshold:
                types.append(TypeEdge.BORDER)
            else:
                types.append(TypeEdge.UNDEFINED)

        return types

    def _fallback_signature_from_polygon(cnt_local):
        """
        Geometric fallback when peak-based corner detection fails.
        """
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

        # conservative fallback typing
        types_local = _classify_fallback_edges(edges_local)

        return np.array(corner_idx, dtype=int), edges_local, types_local

    edges = []
    types_pieces = []
    best_fit = None
    offset = 0

    sigma = 5
    max_sigma = 12
    if not green:
        sigma = 5
        max_sigma = 15

    while sigma <= max_sigma:
        print("Smooth curve with sigma={}...".format(sigma))

        tmp_combs_final = []

        cnt_convert = [c[0] for c in cnt]
        relative_angles = get_relative_angles(
            np.array(cnt_convert), export=False, sigma=sigma
        )
        relative_angles = np.array(relative_angles)
        relative_angles_inverse = -np.array(relative_angles)

        if relative_angles.size == 0:
            sigma += 1
            continue

        max_ra = float(np.max(relative_angles))
        max_rai = float(np.max(relative_angles_inverse))

        extr_tmp = detect_peaks(relative_angles, mph=0.3 * max_ra)
        relative_angles = np.roll(relative_angles, int(len(relative_angles) / 2))
        extr_tmp = np.append(
            extr_tmp,
            (
                detect_peaks(relative_angles, mph=0.3 * float(np.max(relative_angles)))
                - int(len(relative_angles) / 2)
            )
            % len(relative_angles),
            axis=0,
        )
        relative_angles = np.roll(relative_angles, -int(len(relative_angles) / 2))
        extr_tmp = np.unique(extr_tmp)

        extr_tmp_inverse = detect_peaks(relative_angles_inverse, mph=0.3 * max_rai)
        relative_angles_inverse = np.roll(
            relative_angles_inverse, int(len(relative_angles_inverse) / 2)
        )
        extr_tmp_inverse = np.append(
            extr_tmp_inverse,
            (
                detect_peaks(
                    relative_angles_inverse,
                    mph=0.3 * float(np.max(relative_angles_inverse)),
                )
                - int(len(relative_angles_inverse) / 2)
            )
            % len(relative_angles_inverse),
            axis=0,
        )
        extr_tmp_inverse = np.unique(extr_tmp_inverse)

        extr = extr_tmp
        extr_inverse = extr_tmp_inverse

        if len(extr) < 4:
            sigma += 1
            continue

        relative_angles = normalized(relative_angles[:, np.newaxis], axis=0).ravel()

        combs = itertools.permutations(extr, 4)
        combs_l = list(combs)
        OFFSET_LOW = len(relative_angles) / 8
        OFFSET_HIGH = len(relative_angles) / 2.0

        for comb in combs_l:
            if (
                (comb[0] > comb[1])
                and (comb[1] > comb[2])
                and (comb[2] > comb[3])
                and ((comb[0] - comb[1]) > OFFSET_LOW)
                and ((comb[0] - comb[1]) < OFFSET_HIGH)
                and ((comb[1] - comb[2]) > OFFSET_LOW)
                and ((comb[1] - comb[2]) < OFFSET_HIGH)
                and ((comb[2] - comb[3]) > OFFSET_LOW)
                and ((comb[2] - comb[3]) < OFFSET_HIGH)
                and ((comb[3] + (len(relative_angles) - comb[0])) > OFFSET_LOW)
                and ((comb[3] + (len(relative_angles) - comb[0])) < OFFSET_HIGH)
            ):
                candidate = (comb[3], comb[2], comb[1], comb[0])
                if is_acceptable_comb(candidate, extr, len(relative_angles)) and is_acceptable_comb(
                    candidate, extr_inverse, len(relative_angles)
                ):
                    tmp_combs_final.append(candidate)

        sigma += 1
        if len(tmp_combs_final) == 0:
            continue

        best_fit = np.array(
            tmp_combs_final[
                compute_comp(tmp_combs_final, relative_angles, method="flat")
            ],
            dtype=int,
        )

        offset = len(relative_angles) - best_fit[3] - 1
        relative_angles = np.roll(relative_angles, offset)
        best_fit = best_fit + offset
        extr = (extr + offset) % len(relative_angles)
        extr_inverse = (extr_inverse + offset) % len(relative_angles)

        tmp_types_pieces = []
        no_undefined = True
        for best_comb in [
            [0, best_fit[0]],
            [best_fit[0], best_fit[1]],
            [best_fit[1], best_fit[2]],
            [best_fit[2], best_fit[3]],
        ]:
            pos_peaks_inside = peaks_inside(best_comb, extr)
            neg_peaks_inside = peaks_inside(best_comb, extr_inverse)
            pos_peaks_inside.sort()
            neg_peaks_inside.sort()

            t = type_peak(pos_peaks_inside, neg_peaks_inside)
            tmp_types_pieces.append(t)
            if t == TypeEdge.UNDEFINED:
                no_undefined = False

        types_pieces = tmp_types_pieces

        if no_undefined:
            break

    # fallback instead of crash
    if best_fit is None or len(types_pieces) != 4:
        print("[Extractor] corner signature fallback used")
        return _fallback_signature_from_polygon(cnt)

    best_fit_tmp = np.mod(best_fit - offset, len(cnt))

    for i in range(3):
        edges.append(cnt[best_fit_tmp[i] : best_fit_tmp[i + 1]])
    edges.append(
        np.concatenate((cnt[best_fit_tmp[3] :], cnt[: best_fit_tmp[0]]), axis=0)
    )

    edges = [np.array([x[0] for x in e]) for e in edges]
    types_pieces.append(types_pieces[0])
    return best_fit, edges, types_pieces[1:]



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
