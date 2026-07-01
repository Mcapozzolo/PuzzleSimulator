import numpy as np
from .. import config
from .utils import rotate, angle_between


def stick_pieces(bloc_e, p, e, final_stick=False, log_fn=None, centroid_bloc=None, centroid_cand=None):
    """
    Stick an edge of a piece to the bloc of already resolved pieces.

    This keeps the old PREN1 final_stick/TRANSFORM_REPORT behaviour, but also
    supports the SmallSolver/Distance offset-edge alignment via centroid_bloc and
    centroid_cand.
    """
    piece_name = getattr(p, "id", getattr(p, "name", repr(p)))

    if config.EDGE_OFFSET > 0 and centroid_bloc is not None and centroid_cand is not None and hasattr(bloc_e, "compute_offset_shape"):
        bloc_ref = bloc_e.compute_offset_shape(config.EDGE_OFFSET, centroid_bloc)
        e_ref = e.compute_offset_shape(config.EDGE_OFFSET, centroid_cand)
    else:
        bloc_ref = np.asarray(bloc_e.shape, dtype=float)
        e_ref = np.asarray(e.shape, dtype=float)

    translation = np.round(np.subtract(bloc_ref[0], e_ref[-1])).astype(int)
    dx = int(translation[1])
    dy = int(translation[0])

    vec_bloc = np.subtract(bloc_ref[0], bloc_ref[-1])
    vec_piece = np.subtract(e_ref[0], e_ref[-1])
    angle = angle_between((vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0))
    angle_deg = float(np.degrees(angle))
    rot_center = bloc_ref[0]

    for edge in p.edges_:
        edge.shape = edge.shape + translation

    for edge in p.edges_:
        for i, point in enumerate(edge.shape):
            edge.shape[i] = rotate(point, -angle, rot_center)

    # Symmetric endpoint correction, only for non-final geometry matching.
    if not final_stick:
        e_ref_start_translated = e_ref[0] + translation
        e_ref_start_rotated = np.array(rotate((float(e_ref_start_translated[0]), float(e_ref_start_translated[1])), -angle, rot_center))
        gap = np.subtract(bloc_ref[-1], e_ref_start_rotated)
        correction = np.round(gap / 2.0).astype(int)
        for edge in p.edges_:
            edge.shape = edge.shape + correction
        return

    # Old simulator/debug behaviour: transform pixels and log TRANSFORM_REPORT.
    b_e0, b_e1 = bloc_ref[0][0], bloc_ref[0][1]
    minX0, minY0, maxX0, maxY0 = p.get_bbox()
    cx_old = int((minX0 + maxX0) / 2)
    cy_old = int((minY0 + maxY0) / 2)

    p.translate(dx, dy)
    cx_new = cx_old + dx
    cy_new = cy_old + dy

    minX, minY, maxX, maxY = p.get_bbox()
    minX2, minY2, maxX2, maxY2 = p.rotate_bbox(angle, (b_e1, b_e0))
    img_p = p.get_image()

    pixels = {}
    for px in range(minX2, maxX2 + 1):
        for py in range(minY2, maxY2 + 1):
            qx, qy = rotate((px, py), -angle, (b_e1, b_e0))
            qx, qy = int(qx), int(qy)
            if minX <= qx <= maxX and minY <= qy <= maxY and img_p[qx - minX, qy - minY][0] != -1:
                pixels[(px, py)] = img_p[qx - minX, qy - minY]
    p.pixels = pixels

    if log_fn:
        log_fn(f"TRANSFORM_REPORT {piece_name} {cx_old} {cy_old} {cx_new} {cy_new} {dx} {dy} {angle_deg:.1f}")
