import math
import cv2
import scipy.ndimage

import numpy as np
from numba import njit

from .. import config
from .Enums import TypeEdge


@njit(cache=True)
def rgb2hsl(r, g, b):
    r /= 255
    g /= 255
    b /= 255
    minimum = min(r, g, b)
    maximum = max(r, g, b)
    med = (maximum + minimum) / 2
    h, l, s = med, med, med
    if maximum == minimum:
        return 0, 0, l

    d = maximum - minimum
    s = d / (2 - maximum - minimum) if l > 0.5 else d / (maximum + minimum)
    if maximum == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif maximum == g:
        h = (b - r) / d + 2
    elif maximum == b:
        h = (r - g) / d + 4
    return h / 6, s, l


@njit(cache=True)
def rgb2lab(r, g, b, drop_l=False):
    exp = 1 / 3
    r /= 255
    r = (((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92) * 100
    g = g / 255
    g = (((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92) * 100
    b = b / 255
    b = (((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92) * 100
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    x = round(x, 4) / 95.047
    x = x**exp if x > 0.008856 else (7.787 * x) + (16.0 / 116.0)
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    y = round(y, 4) / 100.0
    y = y**exp if y > 0.008856 else (7.787 * y) + (16.0 / 116.0)
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    z = round(z, 4) / 108.883
    z = z**exp if z > 0.008856 else (7.787 * z) + (16.0 / 116.0)
    L = (116.0 * y) - 16
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return round(L, 4) if not drop_l else 0.0, round(a, 4), round(b, 4)


@njit(cache=True)
def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@njit(cache=True)
def dist_edge(e1_begin, e1_end, e2_begin, e2_end):
    dist_e1 = dist(e1_begin, e1_end)
    dist_e2 = dist(e2_begin, e2_end)
    res = math.fabs(dist_e1 - dist_e2)
    val = (dist_e1 + dist_e2) / 2
    return res, val


@njit(cache=True)
def have_edges_similar_length(e1_begin, e1_end, e2_begin, e2_end, percent):
    res, val = dist_edge(e1_begin, e1_end, e2_begin, e2_end)
    return res < (val * percent)


def normalize_vect_len(e1, e2):
    longest = e1 if len(e1) > len(e2) else e2
    shortest = e2 if len(e1) > len(e2) else e1
    return shortest, longest


def diff_match_edges(e1, e2, reverse=True):
    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1 = longest[j]
        x2 = shortest[len(shortest) - i - 1] if reverse else shortest[i]
        diff += (x2 - x1) ** 2
    return diff / len(shortest)


def diff_match_edges2(e1, e2, reverse=True, thres=5, pad=False):
    # Kept for compatibility with old code paths.
    if e2.shape[0] > e1.shape[0]:
        e1, e2 = e2, e1
    if pad:
        pad_length = (e1.shape[0] - e2.shape[0]) // 2
        pad_left = pad_length
        pad_right = pad_length if pad_length * 2 == (e1.shape[0] - e2.shape[0]) else pad_length + 1
        e2 = np.pad(e2, ((pad_left, pad_right), (0, 0)), "constant", constant_values=(0, 0))
    else:
        e1 = e1[: e2.shape[0]]
    if reverse:
        e2 = np.flip(e2, 0)
    best_score = float("inf")
    best_angle = 0
    for angle in range(0, 360):
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        e2_rot = cv2.transform(np.array([e2]), M)[0]
        d = np.linalg.norm(e1 - e2_rot, axis=1)
        score = np.sum(d > thres) / e1.shape[0]
        if score < best_score:
            best_score = score
            best_angle = angle
    return best_score, best_angle


@njit(cache=True)
def dist_color(t1_0, t1_1, t1_2, t2_0, t2_1, t2_2):
    return np.sqrt((t1_0 - t2_0) ** 2 + (t1_1 - t2_1) ** 2 + (t1_2 - t2_2) ** 2)


def euclidean_distance(e1_lab_colors, e2_lab_colors):
    len1 = len(e1_lab_colors)
    len2 = len(e2_lab_colors)
    maximum = max(len1, len2)
    t1 = len1 / maximum
    t2 = len2 / maximum
    return sum([dist_color(*e1_lab_colors[int(t1 * i)], *e2_lab_colors[int(t2 * i)]) for i in range(maximum)])


@njit(cache=True)
def hue2rgb(p, q, t):
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


@njit(cache=True)
def hsl2rgb(h, s, l):
    if s == 0:
        return l, l, l
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (hue2rgb(p, q, h + 1.0 / 3) * 255, hue2rgb(p, q, h) * 255, hue2rgb(p, q, h - 1.0 / 3) * 255)


def get_colors(edge):
    if edge.color is None:
        return []
    return [rgb2lab(*hsl2rgb(col[0], col[1], col[2]), drop_l=True) for col in edge.color]


def _arc_length(pts):
    p = np.asarray(pts, dtype=np.float32)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _resample_curve(points, n_points=100):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts[:1], n_points, axis=0)
    deltas = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    s = np.concatenate([[0], np.cumsum(seg_len)])
    total = s[-1]
    if total == 0:
        return np.repeat(pts[:1], n_points, axis=0)
    t = np.linspace(0, total, n_points)
    resampled = []
    j = 0
    for tt in t:
        while j + 1 < len(s) and s[j + 1] < tt:
            j += 1
        if j + 1 >= len(pts):
            resampled.append(pts[-1])
        else:
            ratio = (tt - s[j]) / (s[j + 1] - s[j] + 1e-8)
            resampled.append(pts[j] + ratio * (pts[j + 1] - pts[j]))
    return np.asarray(resampled, dtype=np.float32)


def _edge_curvature_profile(pts, sigma=2, n_points=50):
    p = np.asarray(pts, dtype=np.float32)
    if len(p) < 3:
        return None
    deltas = np.diff(p, axis=0)
    angles = np.arctan2(-deltas[:, 1], deltas[:, 0])
    angles = np.unwrap(angles)
    curvature = np.diff(angles)
    seg_lens = np.linalg.norm(deltas, axis=1)
    arc_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = arc_s[-1]
    if total < 1e-6:
        return None
    curv_pos = (arc_s[:-2] + arc_s[1:-1] + arc_s[2:]) / 3.0
    t = np.linspace(0.0, total, n_points)
    resampled = np.interp(t, curv_pos, curvature, left=0.0, right=0.0)
    return scipy.ndimage.gaussian_filter(resampled, sigma=sigma)


def _edge_curvature_score(pts1, pts2, n_points=50):
    c1 = _edge_curvature_profile(pts1, n_points=n_points)
    c2 = _edge_curvature_profile(pts2, n_points=n_points)
    if c1 is None or c2 is None:
        return float("inf")
    return float(np.mean(np.abs(c1 + c2[::-1])))


def _overlap_residual(pts1, pts2, n_points=64):
    a = _resample_curve(pts1, n_points=n_points)
    b = _resample_curve(pts2, n_points=n_points)
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    d_rev = float(np.linalg.norm(a - b[::-1], axis=1).mean())
    d_fwd = float(np.linalg.norm(a - b, axis=1).mean())
    return min(d_rev, d_fwd)


def _type_priority_offset(e1, e2):
    t1, t2 = e1.type, e2.type
    complementary = (t1 == TypeEdge.HOLE and t2 == TypeEdge.HEAD) or (t1 == TypeEdge.HEAD and t2 == TypeEdge.HOLE)
    if complementary:
        return 0.0
    if t1 == TypeEdge.UNDEFINED or t2 == TypeEdge.UNDEFINED:
        return 2000.0
    return 3000.0


def _geom_match_score(s1, s2, e1, e2):
    return (
        config.MATCH_RESIDUAL_WEIGHT * _overlap_residual(s1, s2)
        + config.MATCH_CURVATURE_WEIGHT * _edge_curvature_score(s1, s2)
        + _type_priority_offset(e1, e2)
    )


def _offset_shape_for(edge, centroid):
    if config.EDGE_OFFSET > 0 and centroid is not None and hasattr(edge, "compute_offset_shape"):
        return edge.compute_offset_shape(config.EDGE_OFFSET, centroid)
    return edge.shape


def real_edge_compute(e1, e2, centroid1=None, centroid2=None):
    s1 = _offset_shape_for(e1, centroid1)
    s2 = _offset_shape_for(e2, centroid2)
    L1 = _arc_length(s1)
    L2 = _arc_length(s2)
    if abs(L1 - L2) / max(L1, L2, 1.0) > 0.20:
        return float("inf")
    if len(s1) >= 2 and len(s2) >= 2:
        if not have_edges_similar_length(tuple(s1[0]), tuple(s1[-1]), tuple(s2[0]), tuple(s2[-1]), 0.15):
            return float("inf")
    return _geom_match_score(s1, s2, e1, e2)


def generated_edge_compute(e1, e2, centroid1=None, centroid2=None):
    return real_edge_compute(e1, e2, centroid1=centroid1, centroid2=centroid2)


def _edge_geom_distance(e1, e2, n_points=100):
    pts1 = np.asarray(e1.shape, dtype=np.float32)
    pts2 = np.asarray(e2.shape, dtype=np.float32)
    if len(pts1) < 2 or len(pts2) < 2:
        return float("inf")
    pts1 = pts1 - pts1.mean(axis=0, keepdims=True)
    pts2 = pts2 - pts2.mean(axis=0, keepdims=True)
    c1 = _resample_curve(pts1, n_points=n_points)
    c2 = _resample_curve(pts2, n_points=n_points)
    return min(float(np.linalg.norm(c1 - c2, axis=1).mean()), float(np.linalg.norm(c1 - c2[::-1], axis=1).mean()))
