import numpy as np
from .utils import rotate, angle_between


def stick_pieces(bloc_e, p, e, final_stick=False, log_fn=None):
    """
    Stick an edge of a piece to the bloc of already resolved pieces

    :param bloc_e: bloc of edges already solved
    :param p: piece to add to the bloc
    :param e: edge to stick
    :param log_fn: optional callable(str) to append log messages to the GUI
    """

    # ---------- lesbarer Stückname ----------
    piece_name = getattr(p, "id", getattr(p, "name", repr(p)))

    # ---------- 1. Translation-Vektor ----------
    translation = np.subtract(bloc_e.shape[0], e.shape[-1])  # [dy, dx]
    dx = int(translation[1])
    dy = int(translation[0])

    # ---------- 2. Winkel zwischen den Kanten ----------
    vec_bloc = np.subtract(bloc_e.shape[0], bloc_e.shape[-1])
    vec_piece = np.subtract(e.shape[0], e.shape[-1])
    angle = angle_between(
        (vec_bloc[0], vec_bloc[1], 0),
        (-vec_piece[0], -vec_piece[1], 0),
    )
    angle_deg = float(np.degrees(angle))

    # ---------- 3. Kanten-Geometrie verschieben ----------
    for edge in p.edges_:
        edge.shape += translation

    # ---------- 4. Kanten-Geometrie rotieren ----------
    for edge in p.edges_:
        for i, point in enumerate(edge.shape):
            edge.shape[i] = rotate(point, -angle, bloc_e.shape[0])

    # ---------- 5. Finale Pixel-Transformation ----------
    if final_stick:
        # Rotationsursprung
        b_e0, b_e1 = bloc_e.shape[0][0], bloc_e.shape[0][1]

        # STARTKOORDINATEN (vor Verschiebung) – Mittelpunkt der alten BBox
        minX0, minY0, maxX0, maxY0 = p.get_bbox()
        cx_old = int((minX0 + maxX0) / 2)
        cy_old = int((minY0 + maxY0) / 2)

        # Pixels um dx,dy verschieben
        p.translate(dx, dy)

        # ENDKOORDINATEN nach reiner Translation (ohne Rotation)
        # -> so wie in deinem Beispiel: Bewegung: (alt) -> (neu)
        cx_new = cx_old + dx
        cy_new = cy_old + dy

        # Bounding boxes für die Rotationsberechnung (wie im Original)
        minX, minY, maxX, maxY = p.get_bbox()
        minX2, minY2, maxX2, maxY2 = p.rotate_bbox(angle, (b_e1, b_e0))

        # Bild aus aktuellen Pixeln holen
        img_p = p.get_image()

        # neue Pixel in Zielraum schreiben
        pixels = {}
        for px in range(minX2, maxX2 + 1):
            for py in range(minY2, maxY2 + 1):
                # zurück in Ursprung drehen
                qx, qy = rotate((px, py), -angle, (b_e1, b_e0))
                qx, qy = int(qx), int(qy)
                if (
                    minX <= qx <= maxX
                    and minY <= qy <= maxY
                    and img_p[qx - minX, qy - minY][0] != -1
                ):
                    pixels[(px, py)] = img_p[qx - minX, qy - minY]

        # WICHTIG: jetzt können pixels evtl. leer sein,
        # aber wir rufen KEIN get_bbox() mehr danach auf.
        p.pixels = pixels

        # EINZIGE saubere Protokollzeile:
        # TRANSFORM_REPORT <id> <x_alt> <y_alt> <x_neu> <y_neu> <dx> <dy> <rotation>
        if log_fn:
            log_fn(
                f"TRANSFORM_REPORT {piece_name} "
                f"{cx_old} {cy_old} {cx_new} {cy_new} {dx} {dy} {angle_deg:.1f}"
            )
