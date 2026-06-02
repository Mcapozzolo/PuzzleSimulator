import os
import sys
import cv2
from multiprocessing import freeze_support
import time
import math
import numpy as np

print("[BOOT] run_robot_solver.py wurde gestartet", flush=True)

# =========================
# CONFIG
# =========================

USE_CAMERA = False
CAMERA_INDEX = 1

# Kamera-Capture robuster/schneller machen.
# Auf Windows ist CAP_DSHOW meistens deutlich schneller/stabiler als der Default-MSMF-Backend.
CAMERA_USE_DSHOW_ON_WINDOWS = True
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 3
CAMERA_READ_TIMEOUT_SECONDS = 8.0

DO_HOME_BEFORE_RUN = False
HOME_TIMEOUT_SECONDS = 180.0

IMAGE_PATH = r"assets\Bilder aruco marker\00_camera_input.png"

WORKSPACE_SIZE_PX = (1200, 800)

# WICHTIG:
# Das ist die echte physische Grösse der Fläche zwischen den 4 ArUco-Workspace-Ecken.
# Also nicht nur A4, sondern A4 plus Marker-/Randbereich, falls die Marker ausserhalb A4 liegen.
WORKSPACE_SIZE_MM = (321.0, 262.0)

CROP_MARGIN_RATIO_X = 0.02
CROP_MARGIN_RATIO_Y = 0.02

SAFE_Z_MM = 1.0
PICK_Z_MM = 18.0

PUMP_ON_SETTLE_SECONDS = 1.0
PUMP_OFF_SETTLE_SECONDS = 2.0
Z_CONFIRM_SECONDS = 0.5
PUMP_SETTLE_SECONDS = 0.6

# =========================
# ROBOT KOORDINATEN
# =========================

# Roboterkoordinate der ArUco-Workspace-Ecke A0.
# A0 = obere linke Ecke des gewarpten ArUco-Workspace, NICHT Marker-Mitte.
# Diese Werte einmal manuell messen:
# Roboter auf A0 fahren -> X/Y ablesen -> hier eintragen.
PICK_OFFSET_X_MM = 350.5
PICK_OFFSET_Y_MM = 380.0

PICK_SIGN_X = -1
PICK_SIGN_Y = -1

# A5-Zielfläche direkt über ihre vier Ecken im Roboterkoordinatensystem.
# Reihenfolge im Uhrzeigersinn: oben links, unten links, unten rechts, oben rechts.
# Wichtig: Das sind ROBOTERKOORDINATEN, nicht ArUco-/Bildkoordinaten.
A5_TOP_LEFT_ROBOT = (222.0, 90.0)
A5_BOTTOM_LEFT_ROBOT = (222.0, -30.0)
A5_BOTTOM_RIGHT_ROBOT = (32.0, -30.0)
A5_TOP_RIGHT_ROBOT = (32.0, 90.0)

A5_CENTER_MARGIN_X_MM = 5.0
A5_CENTER_MARGIN_Y_MM = 5.0

# Das A5 liegt im Roboter horizontal/landscape: lange Kante = X, kurze Kante = Y.
# Der Sauger-/Greifpunkt wird relativ zur linken oberen Puzzle-Kante platziert.
# Falls die Greifpunktrotation in der Praxis gespiegelt wirkt, diesen Wert auf -1.0 setzen.
GRIP_OFFSET_ROTATION_SIGN = 1.0

ROBOT_MIN_X_MM = 0.0
ROBOT_MAX_X_MM = 350.0
ROBOT_MIN_Y_MM = 0.0
ROBOT_MAX_Y_MM = 350.0
ROBOT_MIN_Z_MM = 0.0
ROBOT_MAX_Z_MM = 18.0
# Damit das gelöste Puzzle nicht exakt auf der A5-Ecke beginnt,
# sondern etwas nach innen verschoben liegt.

SEND_TO_ROBOT = False
ROBOT_PORT = "COM3"

DEBUG_SAVE = True
USE_SCREW_HOLE_PICK = True

# =========================
# IMPORTS
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.Robot.robot_interface import RobotInterface
from solver.Vision import VisionPipeline
from solver.Puzzle.Puzzle import Puzzle
from solver.Vision.robot_coordinates import RobotCoordinateMapper

DEBUG_DIR = os.path.join(PROJECT_ROOT, "assets", "DEBUG_ROBOT_RUN")
TEMP_DIR = os.path.join(PROJECT_ROOT, "assets", "TEST")

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def clear_debug_output_dir():
    """
    Löscht alte Debug-Bilder vor jedem Run.

    Dadurch enthält assets/DEBUG_ROBOT_RUN nach dem Programmstart nur noch
    Bilder, die wirklich zu diesem aktuellen Durchlauf gehören.
    Unterordner bleiben erhalten, damit nichts Unerwartetes ausserhalb des
    Debug-Outputs gelöscht wird.
    """
    if not DEBUG_SAVE:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)

    deleted = 0
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for filename in os.listdir(DEBUG_DIR):
        path = os.path.join(DEBUG_DIR, filename)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_ext:
            continue

        try:
            os.remove(path)
            deleted += 1
        except OSError as exc:
            print(f"[DEBUG WARN] Konnte alte Debug-Datei nicht löschen: {path} ({exc})")

    print(f"[DEBUG] Alte Debug-Bilder gelöscht: {deleted} Datei(en) aus {DEBUG_DIR}")


def save_debug_image(filename, image):
    """Speichert ein Debug-Bild nur, wenn DEBUG_SAVE aktiv ist."""
    if not DEBUG_SAVE or image is None:
        return

    path = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(path, image)
    print(f"[DEBUG] Saved {path}")


def parse_transform_report(log_line):
    parts = log_line.split()

    if len(parts) != 9 or parts[0] != "TRANSFORM_REPORT":
        return None

    return {
        "piece_id": int(parts[1]),
        "x0": float(parts[2]),
        "y0": float(parts[3]),
        "x1": float(parts[4]),
        "y1": float(parts[5]),
        "rotation_deg": float(parts[8]),
    }


def pick_to_robot(x_mm, y_mm):
    return (
        PICK_OFFSET_X_MM + PICK_SIGN_X * x_mm,
        PICK_OFFSET_Y_MM + PICK_SIGN_Y * y_mm,
    )

def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _vec_len(v):
    return math.hypot(v[0], v[1])


def _vec_unit(v):
    l = _vec_len(v)
    if l <= 1e-9:
        raise ValueError("A5-Achse hat Länge 0")
    return (v[0] / l, v[1] / l)


A5_WIDTH_MM = round(_vec_len(_vec_sub(A5_TOP_RIGHT_ROBOT, A5_TOP_LEFT_ROBOT)), 2)
A5_HEIGHT_MM = round(_vec_len(_vec_sub(A5_BOTTOM_LEFT_ROBOT, A5_TOP_LEFT_ROBOT)), 2)
A5_AXIS_X_UNIT = _vec_unit(_vec_sub(A5_TOP_RIGHT_ROBOT, A5_TOP_LEFT_ROBOT))
A5_AXIS_Y_UNIT = _vec_unit(_vec_sub(A5_BOTTOM_LEFT_ROBOT, A5_TOP_LEFT_ROBOT))


def place_to_robot(x_mm, y_mm):
    """
    Wandelt eine A5-interne Platzkoordinate in echte Roboterkoordinaten um.

    x_mm: Abstand von der linken A5-Kante nach rechts.
    y_mm: Abstand von der oberen A5-Kante nach unten.
    """
    robot_x = (
        A5_TOP_LEFT_ROBOT[0]
        + A5_AXIS_X_UNIT[0] * x_mm
        + A5_AXIS_Y_UNIT[0] * y_mm
    )
    robot_y = (
        A5_TOP_LEFT_ROBOT[1]
        + A5_AXIS_X_UNIT[1] * x_mm
        + A5_AXIS_Y_UNIT[1] * y_mm
    )
    return robot_x, robot_y


def rotate_point_px(x, y, rot_deg):
    """Rotiert einen Punkt/Vektor im Bildkoordinatensystem."""
    rot = rot_deg % 360

    if abs(rot - 0) < 1e-9:
        return x, y
    if abs(rot - 90) < 1e-9:
        return y, -x
    if abs(rot - 180) < 1e-9:
        return -x, -y
    if abs(rot - 270) < 1e-9:
        return -y, x

    a = math.radians(rot_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return x * ca - y * sa, x * sa + y * ca


def rotate_vector_by_piece_rotation(dx, dy, rotation_deg):
    """
    Rotiert den Abstand Greifpunkt -> Referenzpunkt mit der Stückrotation.
    Der Sign-Faktor ist bewusst konfigurierbar, weil Solver-/Servo-Richtung je nach
    Kamera- und C-Achsen-Montage gespiegelt sein kann.
    """
    return rotate_point_px(dx, dy, GRIP_OFFSET_ROTATION_SIGN * rotation_deg)


def get_solution_bbox_px(pieces):
    """Gesamte gelöste Puzzle-Bounding-Box in Solver-Pixelkoordinaten."""
    if not pieces:
        raise ValueError("Keine Puzzleteile für solution_bbox_px vorhanden")

    min_rows = []
    min_cols = []
    max_rows = []
    max_cols = []

    for piece in pieces:
        min_row, min_col, max_row, max_col = piece.get_bbox()
        min_rows.append(float(min_row))
        min_cols.append(float(min_col))
        max_rows.append(float(max_row))
        max_cols.append(float(max_col))

    # Rückgabe in (x/col, y/row), weil danach geometrisch mit X/Y gerechnet wird.
    return (
        min(min_cols),
        min(min_rows),
        max(max_cols),
        max(max_rows),
    )


def bbox_corners_px(bbox):
    min_x, min_y, max_x, max_y = bbox
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]


def get_solution_points_px(pieces):
    """Alle Pixelpunkte der gelösten Puzzle-Anordnung als (x=col, y=row)."""
    points = []

    for piece in pieces:
        for row, col in piece.pixels.keys():
            points.append((float(col), float(row)))

    if not points:
        raise ValueError("Keine Pixelpunkte für solution alignment vorhanden")

    return points


def estimate_solution_horizontal_rotation_deg(solution_points_px):
    """
    Berechnet die Korrekturrotation, damit die lange Kante der gelösten
    Puzzle-Anordnung horizontal liegt.

    Rückgabe: Winkel in Grad, der auf Solver-Koordinaten angewendet wird.
    Beispiel: Lösung liegt +7° schräg -> Rückgabe ca. -7°.
    """
    pts = np.asarray(solution_points_px, dtype=np.float32)

    if pts.shape[0] < 5:
        return 0.0

    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle = rect

    # OpenCV liefert den Winkel der Rechteck-Breite. Für die Puzzle-Ausrichtung
    # wollen wir die lange Seite horizontal legen.
    long_side_angle = float(angle)
    if w < h:
        long_side_angle += 90.0

    # Auf den Bereich [-90, 90] normalisieren, damit nicht unnötig fast 180°
    # gedreht wird.
    while long_side_angle <= -90.0:
        long_side_angle += 180.0
    while long_side_angle > 90.0:
        long_side_angle -= 180.0

    return normalize_rotation_deg(-long_side_angle)


def solution_point_to_a5_mm(x_px, y_px, align_info):
    x_rot, y_rot = rotate_point_px(x_px, y_px, align_info["layout_rot"])
    x_mm = align_info["offset_x"] + (x_rot - align_info["min_x"]) * align_info["scale"]
    y_mm = align_info["offset_y"] + (y_rot - align_info["min_y"]) * align_info["scale"]
    return x_mm, y_mm


def draw_a5_aligned_solution_debug(robot_commands, pieces, align_info):
    """
    Zeichnet die final auf A5 ausgerichteten Puzzleteile plus Place-/Greifpunkte.
    So kann direkt kontrolliert werden, ob die berechneten place_x/place_y-Punkte
    sinnvoll innerhalb der Puzzleformen liegen.
    """
    px_per_mm = 4
    pad = 40
    w = int(A5_WIDTH_MM * px_per_mm) + 2 * pad
    h = int(A5_HEIGHT_MM * px_per_mm) + 2 * pad

    debug = np.full((h, w, 3), 245, dtype=np.uint8)

    cv2.rectangle(
        debug,
        (pad, pad),
        (pad + int(A5_WIDTH_MM * px_per_mm), pad + int(A5_HEIGHT_MM * px_per_mm)),
        (0, 180, 0),
        2,
    )

    cv2.putText(
        debug,
        f"A5 robot frame: TL={A5_TOP_LEFT_ROBOT}, TR={A5_TOP_RIGHT_ROBOT}, BL={A5_BOTTOM_LEFT_ROBOT}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    cmd_by_id = {cmd["piece_id"]: cmd for cmd in robot_commands}
    palette = [
        (210, 210, 210), (230, 215, 170), (190, 220, 250), (220, 200, 245),
        (190, 240, 190), (250, 210, 210), (210, 235, 235), (235, 220, 190),
    ]

    # Zuerst die tatsächlichen Puzzleformen zeichnen
    for idx, piece in enumerate(sorted(pieces, key=lambda p: p.id)):
        pts_mm = []
        for row, col in piece.pixels.keys():
            x_mm, y_mm = solution_point_to_a5_mm(float(col), float(row), align_info)
            px = int(round(pad + x_mm * px_per_mm))
            py = int(round(pad + y_mm * px_per_mm))
            pts_mm.append((px, py))

        if not pts_mm:
            continue

        color = palette[idx % len(palette)]
        for px, py in pts_mm:
            if 0 <= px < debug.shape[1] and 0 <= py < debug.shape[0]:
                debug[py, px] = color

        contour = cv2.convexHull(np.array(pts_mm, dtype=np.int32))
        cv2.polylines(debug, [contour], True, (40, 40, 40), 2, cv2.LINE_AA)

        cx = int(np.mean([p[0] for p in pts_mm]))
        cy = int(np.mean([p[1] for p in pts_mm]))
        cv2.putText(
            debug,
            f"{piece.id}",
            (cx - 8, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (160, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Danach Place-/Greifpunkte darüberlegen
    for cmd in robot_commands:
        x = int(round(pad + cmd["place_x_mm"] * px_per_mm))
        y = int(round(pad + cmd["place_y_mm"] * px_per_mm))

        cv2.circle(debug, (x, y), 7, (0, 0, 255), -1)
        cv2.circle(debug, (x, y), 13, (0, 0, 200), 2)
        cv2.putText(
            debug,
            f"P{cmd['piece_id']} ({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f}) / {cmd['rotation_deg']:.1f}deg [{cmd.get('place_point_source','?')}]",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    save_debug_image("25_a5_aligned_place_coordinates.png", debug)


def draw_a5_alignment_diagnostics(robot_commands, pieces, align_info):
    """
    Stärkere Debug-Version für die A5-Platzierung.

    Zeigt pro Puzzleteil:
    - finalen Placepunkt (rot)
    - direkt auf dem gelösten Puzzle erkanntes Loch / Greifpunkt (magenta Text am roten Punkt implizit)
    - transformierten Solver-Referenzpunkt x1/y1 (gelb)
    - rekonstruierten Placepunkt aus transform_report (orange)
    - BBox-Zentrum des gelösten Teils (cyan)
    - Linien zwischen Referenz-/Fallback-Punkt und finalem Placepunkt
    """
    px_per_mm = 5
    pad = 55
    legend_h = 115
    w = int(A5_WIDTH_MM * px_per_mm) + 2 * pad
    h = int(A5_HEIGHT_MM * px_per_mm) + 2 * pad + legend_h

    debug = np.full((h, w, 3), 245, dtype=np.uint8)

    top = pad + legend_h

    # A5-Rahmen
    cv2.rectangle(
        debug,
        (pad, top),
        (pad + int(A5_WIDTH_MM * px_per_mm), top + int(A5_HEIGHT_MM * px_per_mm)),
        (0, 180, 0),
        2,
    )

    cv2.putText(
        debug,
        f"A5 diagnostic frame | TL={A5_TOP_LEFT_ROBOT}, TR={A5_TOP_RIGHT_ROBOT}, BL={A5_BOTTOM_LEFT_ROBOT}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        f"layout_rot={align_info['layout_rot']:.2f} deg | scale={align_info['scale']:.4f} mm/px | offset=({align_info['offset_x']:.1f}, {align_info['offset_y']:.1f}) mm",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    legend = [
        ((0, 0, 255), 'bullseye', 'Finaler Placepunkt (verwendet)'),
        ((0, 165, 255), 'cross', 'Rekonstruierter Placepunkt aus transform_report'),
        ((0, 255, 255), 'cross', 'Solver-Referenzpunkt x1/y1'),
        ((255, 255, 0), 'diamond', 'BBox-Zentrum des gelösten Teils'),
    ]

    lx = 15
    ly = 72
    for color, shape, text in legend:
        if shape == 'bullseye':
            cv2.circle(debug, (lx, ly), 7, color, -1)
            cv2.circle(debug, (lx, ly), 13, (0, 0, 180), 2)
        elif shape == 'diamond':
            pts = np.array([[lx, ly-8], [lx+8, ly], [lx, ly+8], [lx-8, ly]], dtype=np.int32)
            cv2.polylines(debug, [pts], True, color, 2)
        else:
            cv2.line(debug, (lx-8, ly-8), (lx+8, ly+8), color, 2)
            cv2.line(debug, (lx-8, ly+8), (lx+8, ly-8), color, 2)
        cv2.putText(debug, text, (lx + 20, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30,30,30), 1, cv2.LINE_AA)
        ly += 23

    palette = [
        (210, 210, 210), (230, 215, 170), (190, 220, 250), (220, 200, 245),
        (190, 240, 190), (250, 210, 210), (210, 235, 235), (235, 220, 190),
    ]

    # Puzzleformen
    for idx, piece in enumerate(sorted(pieces, key=lambda p: p.id)):
        pts_mm = []
        for row, col in piece.pixels.keys():
            x_mm, y_mm = solution_point_to_a5_mm(float(col), float(row), align_info)
            px = int(round(pad + x_mm * px_per_mm))
            py = int(round(top + y_mm * px_per_mm))
            pts_mm.append((px, py))

        if not pts_mm:
            continue

        color = palette[idx % len(palette)]
        for px, py in pts_mm:
            if 0 <= px < debug.shape[1] and 0 <= py < debug.shape[0]:
                debug[py, px] = color

        contour = cv2.convexHull(np.array(pts_mm, dtype=np.int32))
        cv2.polylines(debug, [contour], True, (40, 40, 40), 2, cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in pts_mm]))
        cy = int(np.mean([p[1] for p in pts_mm]))
        cv2.putText(debug, f"{piece.id}", (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,0,0), 2, cv2.LINE_AA)

    for cmd in robot_commands:
        # finaler Punkt
        fx = int(round(pad + cmd['place_x_mm'] * px_per_mm))
        fy = int(round(top + cmd['place_y_mm'] * px_per_mm))

        # helper to plot mm points
        def mm_to_px(pt_mm):
            return (int(round(pad + pt_mm[0] * px_per_mm)), int(round(top + pt_mm[1] * px_per_mm)))

        # Solver-Referenzpunkt
        if 'place_ref_mm' in cmd:
            rx, ry = mm_to_px(cmd['place_ref_mm'])
            cv2.line(debug, (rx-7, ry-7), (rx+7, ry+7), (0,255,255), 2)
            cv2.line(debug, (rx-7, ry+7), (rx+7, ry-7), (0,255,255), 2)
            cv2.line(debug, (rx, ry), (fx, fy), (0, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(debug, f"R{cmd['piece_id']}", (rx+6, ry-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,160,160), 1, cv2.LINE_AA)

        # Rekonstruierter Fallback-Punkt
        if 'fallback_place_grip_mm' in cmd:
            ox, oy = mm_to_px(cmd['fallback_place_grip_mm'])
            cv2.line(debug, (ox-7, oy-7), (ox+7, oy+7), (0,165,255), 2)
            cv2.line(debug, (ox-7, oy+7), (ox+7, oy-7), (0,165,255), 2)
            cv2.line(debug, (ox, oy), (fx, fy), (0, 140, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, f"F{cmd['piece_id']}", (ox+6, oy+13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,120,220), 1, cv2.LINE_AA)

        # BBox-Zentrum
        if 'bbox_center_mm' in cmd:
            bx, by = mm_to_px(cmd['bbox_center_mm'])
            diamond = np.array([[bx, by-8], [bx+8, by], [bx, by+8], [bx-8, by]], dtype=np.int32)
            cv2.polylines(debug, [diamond], True, (255,255,0), 2)
            cv2.line(debug, (bx, by), (fx, fy), (220, 220, 0), 1, cv2.LINE_AA)
            cv2.putText(debug, f"B{cmd['piece_id']}", (bx+8, by+2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,0), 1, cv2.LINE_AA)

        # Finaler Placepunkt
        cv2.circle(debug, (fx, fy), 7, (0, 0, 255), -1)
        cv2.circle(debug, (fx, fy), 13, (0, 0, 200), 2)

        label = f"P{cmd['piece_id']} ({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f})"
        cv2.putText(debug, label, (fx + 10, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(debug, f"rot={cmd['rotation_deg']:.1f}deg", (fx + 10, fy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,0,0), 1, cv2.LINE_AA)

    save_debug_image("26_a5_alignment_diagnostics.png", debug)


def draw_robot_debug_overlay(image, robot_commands, mapper, workspace_corners=False):
    debug = image.copy()

    # Pickpunkte anzeigen
    for cmd in robot_commands:
        px = int(
            (cmd["pick_x_mm"] / WORKSPACE_SIZE_MM[0]) * WORKSPACE_SIZE_PX[0]
            - mapper.crop_offset_x
        )
        py = int(
            (cmd["pick_y_mm"] / WORKSPACE_SIZE_MM[1]) * WORKSPACE_SIZE_PX[1]
            - mapper.crop_offset_y
        )

        cv2.circle(debug, (px, py), 12, (0, 0, 255), -1)

        text = (
            f"P{cmd['piece_id']} "
            f"({cmd['pick_x_mm']:.1f}, "
            f"{cmd['pick_y_mm']:.1f})"
        )

        cv2.putText(
            debug,
            text,
            (px + 15, py - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ArUco-Workspace-Ecken in mm anzeigen
    if workspace_corners:
        h, w = debug.shape[:2]
        workspace_w_mm, workspace_h_mm = WORKSPACE_SIZE_MM

        corners = [
            ("A0", 20, 35, 0.0, 0.0),
            ("A1", w - 20, 35, workspace_w_mm, 0.0),
            ("A2", w - 20, h - 20, workspace_w_mm, workspace_h_mm),
            ("A3", 20, h - 20, 0.0, workspace_h_mm),
        ]

        for name, x, y, mm_x, mm_y in corners:
            cv2.circle(debug, (x, y), 12, (255, 0, 0), -1)

            text_x = max(20, min(x + 15, w - 260))
            text_y = max(35, min(y - 15, h - 20))

            cv2.putText(
                debug,
                f"{name} ({mm_x:.0f}, {mm_y:.0f}) mm",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    save_debug_image("robot_pick_coordinates.png", debug)


def build_absolute_solution_canvas(pieces, background_value=255):
    """
    Rendert die gelösten Puzzleteile in ein gemeinsames Bild, dessen Indizes
    direkt zu den absoluten Piece-Koordinaten passen.
    """
    if not pieces:
        return np.full((1, 1, 3), background_value, dtype=np.uint8)

    max_x = 0
    max_y = 0
    for piece in pieces:
        min_x, min_y, max_x_piece, max_y_piece = piece.get_bbox()
        max_x = max(max_x, int(max_x_piece))
        max_y = max(max_y, int(max_y_piece))

    canvas = np.full((max_x + 2, max_y + 2, 3), background_value, dtype=np.uint8)

    for piece in pieces:
        for (x, y), color in piece.pixels.items():
            xi = int(x)
            yi = int(y)
            if 0 <= xi < canvas.shape[0] and 0 <= yi < canvas.shape[1]:
                canvas[xi, yi] = np.asarray(color, dtype=np.uint8)

    return canvas


def detect_pick_centers_on_solved_puzzle(pieces):
    """
    Erkennt die finalen Schraubenloch-/Greifpunkte direkt auf dem GELÖSTEN Puzzle.
    Das ist robuster als die Rekonstruktion über TRANSFORM_REPORT, weil die
    Solver-Logs nur Translation und Winkel liefern, aber nicht den finalen
    absoluten Rotationsursprung.
    """
    solution_img = build_absolute_solution_canvas(pieces)
    debug = solution_img.copy()
    centers = {}

    for piece in pieces:
        center = detect_screw_hole_center(piece, solution_img, debug_img=debug)
        centers[int(piece.id)] = center
        cv2.putText(
            debug,
            f"SP{piece.id}",
            (int(center["col"]) + 10, int(center["row"]) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    save_debug_image("05_solved_detected_screw_holes.png", debug)
    return centers

def detect_screw_hole_center(piece, source_img, debug_img=None):
    """
    Erkennt das Schraubenloch eines Puzzleteils visuell.

    Wichtig:
    - sucht nur innerhalb der Piece-Maske
    - ignoriert Randbereiche des Puzzleteils
    - bevorzugt runde, kleine, helle Strukturen
    - fallback: BBox-Mitte
    """

    min_row, min_col, max_row, max_col = piece.get_bbox()

    min_row = max(0, int(min_row))
    min_col = max(0, int(min_col))
    max_row = min(source_img.shape[0] - 1, int(max_row))
    max_col = min(source_img.shape[1] - 1, int(max_col))

    roi = source_img[min_row:max_row + 1, min_col:max_col + 1].copy()

    if roi.size == 0:
        return {
            "row": (min_row + max_row) / 2,
            "col": (min_col + max_col) / 2,
        }

    piece_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    for row, col in piece.pixels.keys():
        rr = int(row) - min_row
        cc = int(col) - min_col

        if 0 <= rr < piece_mask.shape[0] and 0 <= cc < piece_mask.shape[1]:
            piece_mask[rr, cc] = 255

    # Randbereich entfernen, damit Einbuchtungen/Reflexe an Kanten nicht erkannt werden
    erode_kernel = np.ones((13, 13), np.uint8)
    inner_mask = cv2.erode(piece_mask, erode_kernel, iterations=1)

    if np.count_nonzero(inner_mask) < 50:
        inner_mask = piece_mask.copy()

    dist = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Lokalen Kontrast verbessern
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Helle Metall-/Lochregionen
    bright = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -8,
    )

    bright = cv2.bitwise_and(bright, bright, mask=inner_mask)

    kernel = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        bright,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 8 or area > 400:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.25:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if radius < 2.0 or radius > 14.0:
            continue

        rr = int(round(y))
        cc = int(round(x))

        if not (0 <= rr < piece_mask.shape[0] and 0 <= cc < piece_mask.shape[1]):
            continue

        if inner_mask[rr, cc] == 0:
            continue

        edge_distance = float(dist[rr, cc])

        # Loch sollte nicht direkt an der Puzzlekante liegen
        if edge_distance < 8:
            continue

        local = gray[
            max(0, rr - 5):min(gray.shape[0], rr + 6),
            max(0, cc - 5):min(gray.shape[1], cc + 6),
        ]

        local_brightness = float(np.mean(local)) if local.size else 0.0

        score = (
            circularity * 100.0
            + edge_distance * 4.0
            + local_brightness * 0.2
            - abs(radius - 5.0) * 3.0
        )

        candidates.append({
            "row": min_row + y,
            "col": min_col + x,
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "edge_distance": edge_distance,
            "score": score,
        })

    # Zweiter Versuch: HoughCircles, falls Konturen nichts Gutes finden
    if not candidates:
        masked_gray = cv2.bitwise_and(gray, gray, mask=inner_mask)

        circles = cv2.HoughCircles(
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=80,
            param2=10,
            minRadius=3,
            maxRadius=12,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            for x, y, r in circles:
                if not (0 <= y < piece_mask.shape[0] and 0 <= x < piece_mask.shape[1]):
                    continue

                if inner_mask[y, x] == 0:
                    continue

                edge_distance = float(dist[y, x])

                if edge_distance < 8:
                    continue

                candidates.append({
                    "row": float(min_row + y),
                    "col": float(min_col + x),
                    "radius": float(r),
                    "area": float(np.pi * r * r),
                    "circularity": 1.0,
                    "edge_distance": edge_distance,
                    "score": 200.0 + edge_distance * 4.0,
                })

    if candidates:
        best = max(candidates, key=lambda c: c["score"])

        row = float(best["row"])
        col = float(best["col"])

        if debug_img is not None:
            cv2.circle(debug_img, (int(col), int(row)), 9, (0, 255, 255), -1)
            cv2.putText(
                debug_img,
                f"H{piece.id}",
                (int(col) + 10, int(row) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        print(
            f"[HOLE] Piece {piece.id}: "
            f"row={row:.1f}, col={col:.1f}, "
            f"r={best['radius']:.1f}, "
            f"circ={best['circularity']:.2f}, "
            f"dist={best['edge_distance']:.1f}, "
            f"score={best['score']:.1f}"
        )

        return {
            "row": row,
            "col": col,
        }

    fallback_row = (min_row + max_row) / 2
    fallback_col = (min_col + max_col) / 2

    print(
        f"[HOLE] Piece {piece.id}: kein Loch erkannt -> "
        f"Fallback BBox-Mitte row={fallback_row:.1f}, col={fallback_col:.1f}"
    )

    if debug_img is not None:
        cv2.circle(debug_img, (int(fallback_col), int(fallback_row)), 9, (0, 0, 255), -1)
        cv2.putText(
            debug_img,
            f"FB{piece.id}",
            (int(fallback_col) + 10, int(fallback_row) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "row": fallback_row,
        "col": fallback_col,
    }

def normalize_rotation_deg(angle_deg):
    return round(((angle_deg + 180) % 360) - 180, 2)

def rotation_to_servo_steps(angle_deg):
    angle = normalize_rotation_deg(angle_deg)

    if angle < 0:
        # Beispiel -70:
        # zuerst auf 70, dann auf 0
        return abs(angle), 0.0

    if angle > 0:
        # Beispiel +70:
        # zuerst auf 180-70=110, dann auf 180
        return 180.0 - angle, 180.0

    return 90.0, 90.0

def build_pick_place_sequence(cmd):
    pick_x, pick_y = pick_to_robot(
        cmd["pick_x_mm"],
        cmd["pick_y_mm"],
    )

    place_x, place_y = place_to_robot(
        cmd["place_x_mm"],
        cmd["place_y_mm"],
    )

    pre_rot, final_rot = rotation_to_servo_steps(cmd["rotation_deg"])

    pre_rot = max(2.0, min(178.0, pre_rot))
    final_rot = max(2.0, min(178.0, final_rot))

    return [
        {
            "description": "pre_rotate",
            "type": "rotate",
            "rotation_deg": pre_rot,
        },
        {
            "description": "above_pick",
            "type": "move",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": SAFE_Z_MM,
            "rotation_deg": pre_rot,
        },
        {
            "description": "down_to_pick",
            "type": "move",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": PICK_Z_MM,
            "rotation_deg": pre_rot,
        },
        {
            "description": "suction_on",
            "type": "suction",
        },
        {
            "description": "lift_piece",
            "type": "move",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": SAFE_Z_MM,
            "rotation_deg": pre_rot,
        },
        {
            "description": "final_rotate",
            "type": "rotate",
            "rotation_deg": final_rot,
        },
        {
            "description": "above_place",
            "type": "move",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": SAFE_Z_MM,
            "rotation_deg": final_rot,
        },
        {
            "description": "down_to_place",
            "type": "move",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": PICK_Z_MM,
            "rotation_deg": final_rot,
        },
        {
            "description": "suction_off",
            "type": "suction",
        },
        {
            "description": "lift_after_place",
            "type": "move",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": SAFE_Z_MM,
            "rotation_deg": final_rot,
        },
    ]


def print_robot_commands(robot_commands):
    print("\n[ROBOT COMMANDS]")

    for cmd in robot_commands:
        print(
            f"\nPiece {cmd['piece_id']}: "
            f"Pick=({cmd['pick_x_mm']:.1f}, {cmd['pick_y_mm']:.1f}) mm | "
            f"Place=({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f}) mm | "
            f"Rot={cmd['rotation_deg']:.1f}°"
        )

        for step in build_pick_place_sequence(cmd):

            if step.get("type") == "rotate":
                print(
                    f"  {step['description']}: "
                    f"C={step['rotation_deg']:.2f}°"
                )
                continue

            if step.get("type") == "suction":
                if step["description"] == "suction_on":
                    print(f"  {step['description']}: PUMP=True")
                else:
                    print(f"  {step['description']}: PUMP=False")
                continue

            print(
                f"  {step['description']}: "
                f"X={step['x_mm']:.2f} mm, "
                f"Y={step['y_mm']:.2f} mm, "
                f"Z={step['z_mm']:.2f} mm, "
                f"C={step.get('rotation_deg', 0.0):.2f}°"
            )

def validate_robot_position(x, y, z, c, description=""):
    errors = []

    if not (ROBOT_MIN_X_MM <= x <= ROBOT_MAX_X_MM):
        errors.append(f"X={x:.2f} ausserhalb [{ROBOT_MIN_X_MM}, {ROBOT_MAX_X_MM}]")

    if not (ROBOT_MIN_Y_MM <= y <= ROBOT_MAX_Y_MM):
        errors.append(f"Y={y:.2f} ausserhalb [{ROBOT_MIN_Y_MM}, {ROBOT_MAX_Y_MM}]")

    if not (ROBOT_MIN_Z_MM <= z <= ROBOT_MAX_Z_MM):
        errors.append(f"Z={z:.2f} ausserhalb [{ROBOT_MIN_Z_MM}, {ROBOT_MAX_Z_MM}]")

    if not (2 <= c <= 178):
        errors.append(f"C={c:.2f} ausserhalb [2, 178]")

    if errors:
        raise ValueError(
            f"Ungültige Roboterposition bei {description}: "
            + "; ".join(errors)
        )

def send_to_robot(robot_commands):
    robot = RobotInterface(port=ROBOT_PORT, send_units="cm")

    current_pose = {
        "x_mm": None,
        "y_mm": None,
        "z_mm": None,
        "c_deg": 90,
    }

    def clamp_c_local(c_deg):
        c_deg = float(c_deg)

        if c_deg < 2.0:
            return 2

        if c_deg > 178.0:
            return 178

        return int(round(c_deg))

    def send_move(step, pump=None):
        nonlocal current_pose

        x = step.get("x_mm", current_pose["x_mm"])
        y = step.get("y_mm", current_pose["y_mm"])
        z = step.get("z_mm", current_pose["z_mm"])
        c = clamp_c_local(step.get("rotation_deg", current_pose["c_deg"]))

        if x is None or y is None or z is None:
            raise ValueError(
                f"Kann MOVE nicht senden, weil Position unbekannt ist: {step}"
            )

        validate_robot_position(x, y, z, c, step["description"])

        pump_text = "UNCHANGED" if pump is None else str(bool(pump))

        print(
            f"[ROBOT MOVE] {step['description']} "
            f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}, C={c:.2f}, PUMP={pump_text}"
        )

        robot.move_xyzc_mm_and_wait(
            x,
            y,
            z,
            c,
            pump=pump,
        )

        current_pose = {
            "x_mm": x,
            "y_mm": y,
            "z_mm": z,
            "c_deg": c,
        }

    def send_pump_on():
        if current_pose["x_mm"] is None:
            raise ValueError("PUMP ON ohne bekannte Position nicht möglich.")

        step = {
            "description": "suction_on",
            "x_mm": current_pose["x_mm"],
            "y_mm": current_pose["y_mm"],
            "z_mm": current_pose["z_mm"],
            "rotation_deg": current_pose["c_deg"],
        }

        send_move(step, pump=True)
        robot.wait_for_message_contains("Pump and valve activated", timeout_s=3.0)
        time.sleep(PUMP_ON_SETTLE_SECONDS)

    def send_pump_off():
        if current_pose["x_mm"] is None:
            raise ValueError("PUMP OFF ohne bekannte Position nicht möglich.")

        step = {
            "description": "suction_off",
            "x_mm": current_pose["x_mm"],
            "y_mm": current_pose["y_mm"],
            "z_mm": current_pose["z_mm"],
            "rotation_deg": current_pose["c_deg"],
        }

        # 1. Versuch: Pumpe aus
        send_move(step, pump=False)
        ok = robot.wait_for_message_contains("Pump and valve deactivated", timeout_s=3.0)

        # 2. Versuch, falls Firmware die Deaktivierung nicht bestätigt
        if not ok:
            print("[ROBOT WARN] PUMP OFF nicht bestätigt -> sende PUMP=False nochmals")
            send_move(step, pump=False)
            robot.wait_for_message_contains("Pump and valve deactivated", timeout_s=3.0)

        time.sleep(PUMP_OFF_SETTLE_SECONDS)

    def confirm_z_down(step):
        """
        Sicherheitsbestätigung für down_to_pick/down_to_place.
        Sendet denselben Z-Zielpunkt nochmals, damit die Z-Achse sicher unten ist.
        """
        if step["description"] not in ("down_to_pick", "down_to_place"):
            return

        print(f"[ROBOT] Z-Confirm für {step['description']}")
        send_move(step, pump=None)
        time.sleep(Z_CONFIRM_SECONDS)

    try:
        print("[ROBOT] READY")
        robot.ready()
        robot.wait_until_idle()

        if DO_HOME_BEFORE_RUN:
            print("[ROBOT] HOME")
            robot.home()
            robot.wait_until_idle(timeout_s=HOME_TIMEOUT_SECONDS)
            print("[ROBOT] HOME fertig")
            time.sleep(1.0) 

        for cmd in robot_commands:
            print(f"[ROBOT] Piece {cmd['piece_id']}")

            for step in build_pick_place_sequence(cmd):

                if step.get("type") == "rotate":
                    current_pose["c_deg"] = clamp_c_local(step["rotation_deg"])

                    if current_pose["x_mm"] is None:
                        print(
                            f"[ROBOT ROTATE PRESET] "
                            f"C={current_pose['c_deg']:.2f} wird beim nächsten MOVE mitgesendet"
                        )
                    else:
                        rotate_step = {
                            "description": step["description"],
                            "x_mm": current_pose["x_mm"],
                            "y_mm": current_pose["y_mm"],
                            "z_mm": current_pose["z_mm"],
                            "rotation_deg": current_pose["c_deg"],
                        }
                        send_move(rotate_step, pump=None)

                    continue

                if step.get("type") == "suction":
                    if step["description"] == "suction_on":
                        send_pump_on()
                    elif step["description"] == "suction_off":
                        send_pump_off()
                    else:
                        raise ValueError(f"Unbekannter suction step: {step}")
                    continue

                send_move(step, pump=None)
                confirm_z_down(step)

        # Sicherheit: am Schluss Pumpe aus
        if current_pose["x_mm"] is not None:
            send_pump_off()

        print("[ROBOT] FINISH")
        robot.finish()

    finally:
        robot.close()


def align_solution_to_a5(robot_commands, solution_points_px):
    """
    Richtet das vom Solver gelöste Puzzle horizontal in die A5-Fläche ein.

    Wichtig: Platziert wird nicht mehr der Solver-Referenzpunkt/Mittelpunkt,
    sondern der tatsächliche Greifpunkt. Dafür wird der Abstand vom Greifpunkt
    zur linken/oberen Puzzlekante berechnet und auf die linke/obere A5-Kante
    plus Rand übertragen.
    """
    if not robot_commands:
        return None

    target_w = A5_WIDTH_MM - 2 * A5_CENTER_MARGIN_X_MM
    target_h = A5_HEIGHT_MM - 2 * A5_CENTER_MARGIN_Y_MM

    # 1) Für jedes Teil den Zielpunkt des Saugers in der gelösten Solver-Ebene berechnen.
    for cmd in robot_commands:
        solved_pick_px = cmd.get("solved_pick_px")

        pick_x, pick_y = cmd["pick_px"]
        src_x, src_y = cmd["source_ref_px"]
        place_ref_x, place_ref_y = cmd["raw_place_ref_px"]

        grip_dx = pick_x - src_x
        grip_dy = pick_y - src_y

        grip_dx_rot, grip_dy_rot = rotate_vector_by_piece_rotation(
            grip_dx,
            grip_dy,
            cmd["rotation_deg"],
        )

        fallback_grip_px = (
            place_ref_x + grip_dx_rot,
            place_ref_y + grip_dy_rot,
        )

        cmd["fallback_place_grip_px"] = fallback_grip_px
        cmd["place_ref_px"] = (place_ref_x, place_ref_y)

        if solved_pick_px is not None:
            # Beste Variante: Schraubenloch/Greifpunkt direkt auf dem gelösten Puzzle erkennen.
            cmd["raw_place_grip_px"] = solved_pick_px
            cmd["place_point_source"] = "solved_hole_detection"
        else:
            # Fallback: über Solver-Referenzpunkt rekonstruieren.
            cmd["raw_place_grip_px"] = fallback_grip_px
            cmd["place_point_source"] = "transform_report_fallback"

    # 2) Nicht nur 0/90/180/270 testen: zuerst die echte Schräglage der
    #    gelösten Puzzle-Pixel bestimmen und diese herausdrehen.
    base_deskew_deg = estimate_solution_horizontal_rotation_deg(solution_points_px)

    candidates = []

    # 0/180 behalten die lange Seite horizontal; +90/+270 sind Fallbacks, falls
    # das Puzzle im A5 doch hochkant besser passt.
    for extra_rot in [0, 180, 90, 270]:
        layout_rot = normalize_rotation_deg(base_deskew_deg + extra_rot)

        rotated_solution_points = [
            rotate_point_px(x, y, layout_rot)
            for x, y in solution_points_px
        ]

        min_x = min(p[0] for p in rotated_solution_points)
        min_y = min(p[1] for p in rotated_solution_points)
        max_x = max(p[0] for p in rotated_solution_points)
        max_y = max(p[1] for p in rotated_solution_points)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        scale = min(target_w / bbox_w, target_h / bbox_h)
        final_w = bbox_w * scale
        final_h = bbox_h * scale

        # A5 ist horizontal: bevorzugt wird eine horizontale/landscape Lösung.
        landscape_penalty = 0 if final_w >= final_h else 10_000
        unused_area = (target_w * target_h) - (final_w * final_h)

        # Kleine Zusatzrotationen um 180° sind okay, 90° nur wenn wirklich nötig.
        extra_rotation_penalty = 0 if extra_rot in [0, 180] else 1_000

        score = landscape_penalty + extra_rotation_penalty + unused_area * 0.01

        candidates.append({
            "score": score,
            "layout_rot": layout_rot,
            "base_deskew_deg": base_deskew_deg,
            "extra_rot": extra_rot,
            "min_x": min_x,
            "min_y": min_y,
            "scale": scale,
            "final_w": final_w,
            "final_h": final_h,
        })

    if not candidates:
        raise ValueError("A5 alignment failed: no valid placement candidate")

    best = min(candidates, key=lambda c: c["score"])

    offset_x = A5_CENTER_MARGIN_X_MM + (target_w - best["final_w"]) / 2
    offset_y = A5_CENTER_MARGIN_Y_MM + (target_h - best["final_h"]) / 2

    for cmd in robot_commands:
        grip_x, grip_y = cmd["raw_place_grip_px"]
        grip_x_rot, grip_y_rot = rotate_point_px(grip_x, grip_y, best["layout_rot"])

        # Abstand vom Greifpunkt zur linken/oberen horizontalisierten Puzzlekante.
        grip_dist_from_left_mm = (grip_x_rot - best["min_x"]) * best["scale"]
        grip_dist_from_top_mm = (grip_y_rot - best["min_y"]) * best["scale"]

        # Dieser Abstand wird auf die A5-Kante + Zentrier-Offset übertragen.
        cmd["place_x_mm"] = offset_x + grip_dist_from_left_mm
        cmd["place_y_mm"] = offset_y + grip_dist_from_top_mm

        # Die globale Puzzle-Ausrichtung muss auch in die Teilrotation einfliessen.
        cmd["rotation_deg"] = normalize_rotation_deg(
            cmd["rotation_deg"] + best["layout_rot"]
        )

        cmd["a5_grip_offset_from_left_mm"] = grip_dist_from_left_mm
        cmd["a5_grip_offset_from_top_mm"] = grip_dist_from_top_mm

        if "place_ref_px" in cmd:
            cmd["place_ref_mm"] = solution_point_to_a5_mm(
                cmd["place_ref_px"][0],
                cmd["place_ref_px"][1],
                {"layout_rot": best["layout_rot"], "offset_x": offset_x, "offset_y": offset_y, "min_x": best["min_x"], "min_y": best["min_y"], "scale": best["scale"]}
            )

        if "fallback_place_grip_px" in cmd:
            cmd["fallback_place_grip_mm"] = solution_point_to_a5_mm(
                cmd["fallback_place_grip_px"][0],
                cmd["fallback_place_grip_px"][1],
                {"layout_rot": best["layout_rot"], "offset_x": offset_x, "offset_y": offset_y, "min_x": best["min_x"], "min_y": best["min_y"], "scale": best["scale"]}
            )

        if "solved_bbox_center_px" in cmd:
            cmd["bbox_center_mm"] = solution_point_to_a5_mm(
                cmd["solved_bbox_center_px"][0],
                cmd["solved_bbox_center_px"][1],
                {"layout_rot": best["layout_rot"], "offset_x": offset_x, "offset_y": offset_y, "min_x": best["min_x"], "min_y": best["min_y"], "scale": best["scale"]}
            )

    best["offset_x"] = offset_x
    best["offset_y"] = offset_y

    print(
        f"[A5 ALIGN] A5_size=({A5_WIDTH_MM:.1f}, {A5_HEIGHT_MM:.1f}) mm, "
        f"deskew={best['base_deskew_deg']:.2f}°, "
        f"extra={best['extra_rot']}°, "
        f"layout_rot={best['layout_rot']:.2f}°, "
        f"scale={best['scale']:.4f} mm/px, "
        f"puzzle_size=({best['final_w']:.1f}, {best['final_h']:.1f}) mm, "
        f"offset=({offset_x:.1f}, {offset_y:.1f}) mm"
    )

    return best


def log_duration(label, start_time):
    print(f"[TIME] {label}: {time.perf_counter() - start_time:.2f}s", flush=True)


def capture_camera_frame():
    """
    Schneller Kamera-Snapshot mit Diagnoseausgaben.

    Der alte Code verwendete cv2.VideoCapture(CAMERA_INDEX) ohne Backend und machte
    danach genau ein read(). Auf Windows kann der Default-Backend stark verzögern.
    Mit CAP_DSHOW, fixer Auflösung, MJPG und wenigen Warmup-Frames ist der Snapshot
    normalerweise viel schneller und reproduzierbarer.
    """
    backend = cv2.CAP_ANY
    backend_name = "CAP_ANY"

    if os.name == "nt" and CAMERA_USE_DSHOW_ON_WINDOWS:
        backend = cv2.CAP_DSHOW
        backend_name = "CAP_DSHOW"

    print(
        f"[CAM] Öffne Kamera index={CAMERA_INDEX}, backend={backend_name}...",
        flush=True,
    )

    t_open = time.perf_counter()
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    log_duration("camera open", t_open)

    if not cap.isOpened() and backend != cv2.CAP_ANY:
        print("[CAM WARN] CAP_DSHOW konnte Kamera nicht öffnen -> Fallback CAP_ANY", flush=True)
        cap.release()
        t_open = time.perf_counter()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
        log_duration("camera open fallback", t_open)

    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera konnte nicht geöffnet werden. Prüfe CAMERA_INDEX={CAMERA_INDEX}."
        )

    # Einige Backends ignorieren diese Werte, aber wenn sie greifen, reduzieren sie
    # Verzögerung und verhindern unnötig grosse Frames.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    frame = None
    deadline = time.perf_counter() + CAMERA_READ_TIMEOUT_SECONDS

    # Mehr als ein Frame lesen, damit nicht ein alter/unterbelichteter Startframe kommt.
    for i in range(max(1, CAMERA_WARMUP_FRAMES + 1)):
        if time.perf_counter() > deadline:
            break

        t_read = time.perf_counter()
        ret, candidate = cap.read()
        print(
            f"[CAM] read {i + 1}/{CAMERA_WARMUP_FRAMES + 1}: "
            f"ret={ret}, dt={time.perf_counter() - t_read:.2f}s",
            flush=True,
        )

        if ret and candidate is not None:
            frame = candidate

    cap.release()

    if frame is None:
        raise RuntimeError("Kamera konnte innerhalb des Timeouts kein Bild aufnehmen.")

    print(f"[CAM] Frame shape={frame.shape}", flush=True)
    return frame


def main():
    clear_debug_output_dir()

    t_total = time.perf_counter()
    print("[RUN] Starte Vision Pipeline...", flush=True)

    pipeline = VisionPipeline(
        marker_length_mm=30.0,
        workspace_output_size_px=WORKSPACE_SIZE_PX,
        workspace_mm_size=WORKSPACE_SIZE_MM,
        aruco_ids=(0, 1, 2, 3),
    )

    if USE_CAMERA:
        t_cam = time.perf_counter()
        frame = capture_camera_frame()
        log_duration("camera total", t_cam)

        save_debug_image("00_camera_input.png", frame)

        t_vision = time.perf_counter()
        result = pipeline.process_image(frame)
        log_duration("pipeline.process_image(camera frame)", t_vision)
    else:
        t_img = time.perf_counter()
        input_image = cv2.imread(IMAGE_PATH)
        if input_image is None:
            raise RuntimeError(f"Bild konnte nicht geladen werden: {IMAGE_PATH}")
        log_duration("cv2.imread", t_img)

        save_debug_image("00_camera_input.png", input_image)

        t_vision = time.perf_counter()
        result = pipeline.process_image(input_image)
        log_duration("pipeline.process_image(file image)", t_vision)

    warped = result["warped_workspace"]
    if warped is None:
        raise ValueError("Warped workspace ist None -> ArUco Fehler")

    h, w = warped.shape[:2]

    margin_x = int(w * CROP_MARGIN_RATIO_X)
    margin_y = int(h * CROP_MARGIN_RATIO_Y)

    warped_inner = warped[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ].copy()

    warped_path = os.path.join(TEMP_DIR, "robot_solver_input.png")
    cv2.imwrite(warped_path, warped_inner)

    save_debug_image("01_aruco_debug.png", result["aruco_debug"])
    save_debug_image("02_warped_workspace.png", warped)
    save_debug_image("03_warped_inner.png", warped_inner)

    transformation_logs = []

    def record_log(msg):
        print(msg)

        if msg.startswith("TRANSFORM_REPORT"):
            transformation_logs.append(msg)

    puzzle = Puzzle(warped_path, log_fn=record_log)
    print("[DEBUG] warped shape:", warped.shape)
    print("[RUN] Extrahiere Puzzleteile...")
    puzzle.extract_pieces()

    print(f"[RUN] Extrahierte Teile: {len(puzzle.pieces_)}")

    initial_pick_centers = {}

    if USE_SCREW_HOLE_PICK:
        hole_debug = warped_inner.copy()

        for piece in puzzle.pieces_:
            initial_pick_centers[piece.id] = detect_screw_hole_center(
                piece,
                warped_inner,
                debug_img=hole_debug,
            )

        save_debug_image("04_detected_screw_holes.png", hole_debug)

    else:
        fallback_debug = warped_inner.copy()

        for piece in puzzle.pieces_:
            min_row, min_col, max_row, max_col = piece.get_bbox()

            row = (min_row + max_row) / 2
            col = (min_col + max_col) / 2

            initial_pick_centers[piece.id] = {
                "row": row,
                "col": col,
            }

            cv2.circle(fallback_debug, (int(col), int(row)), 9, (0, 0, 255), -1)
            cv2.putText(
                fallback_debug,
                f"FB{piece.id}",
                (int(col) + 10, int(row) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            print(
                f"[PICK] Piece {piece.id}: "
                f"Fallback BBox-Mitte row={row:.1f}, col={col:.1f}"
            )

        save_debug_image("04_detected_screw_holes.png", fallback_debug)

    if len(puzzle.pieces_) == 0:
        raise ValueError("Keine Puzzleteile erkannt!")

    print("[RUN] Löse Puzzle...")
    puzzle.solve_puzzle()
    print("[RUN] Puzzle gelöst.")

    # Aktuelle Solver-Debugbilder speichern. Weil der Debug-Ordner am Start
    # geleert wurde, können hier keine alten 15_debug/18_debug/... Bilder liegen bleiben.
    for i, debug_img in enumerate(puzzle.get_debug_images(), start=10):
        save_debug_image(f"{i:02d}_debug.png", debug_img)

    solved_pick_centers = detect_pick_centers_on_solved_puzzle(puzzle.pieces_)
    solved_piece_stats = {}
    for piece in puzzle.pieces_:
        min_x, min_y, max_x, max_y = piece.get_bbox()
        solved_piece_stats[int(piece.id)] = {
            "bbox_center_px": (float((min_y + max_y) / 2.0), float((min_x + max_x) / 2.0)),
        }

    mapper = RobotCoordinateMapper(
        workspace_size_px=WORKSPACE_SIZE_PX,
        workspace_size_mm=WORKSPACE_SIZE_MM,
        crop_offset_px=(margin_x, margin_y),
    )

    robot_commands = []

    for log in transformation_logs:
        report = parse_transform_report(log)

        if report is None:
            continue

        pick_center = initial_pick_centers.get(report["piece_id"])

        cmd = mapper.transform_report_to_robot_command(
            report,
            pick_center=pick_center,
        )

        # Zusätzliche Pixel-Metadaten für präzise A5-Platzierung:
        # - pick_px ist der reale Greifpunkt im Solver-Crop
        # - source_ref_px ist der ursprüngliche Solver-Referenzpunkt
        # - raw_place_ref_px ist der Solver-Zielreferenzpunkt vor A5-Normalisierung
        if pick_center is not None:
            pick_px = (float(pick_center["col"]), float(pick_center["row"]))
        else:
            pick_px = (float(report["y0"]), float(report["x0"]))

        cmd["pick_px"] = pick_px
        cmd["source_ref_px"] = (float(report["y0"]), float(report["x0"]))
        cmd["raw_place_ref_px"] = (float(report["y1"]), float(report["x1"]))

        solved_pick = solved_pick_centers.get(int(report["piece_id"]))
        if solved_pick is not None:
            cmd["solved_pick_px"] = (float(solved_pick["col"]), float(solved_pick["row"]))

        stats = solved_piece_stats.get(int(report["piece_id"]))
        if stats is not None:
            cmd["solved_bbox_center_px"] = stats["bbox_center_px"]

        robot_commands.append(cmd)

    robot_commands = sorted(
        robot_commands,
        key=lambda x: x["piece_id"],
    )
    solution_points_px = get_solution_points_px(puzzle.pieces_)
    align_info = align_solution_to_a5(robot_commands, solution_points_px)
    draw_a5_aligned_solution_debug(robot_commands, puzzle.pieces_, align_info)
    draw_a5_alignment_diagnostics(robot_commands, puzzle.pieces_, align_info)

    print_robot_commands(robot_commands)

    draw_robot_debug_overlay(
        warped,
        robot_commands,
        mapper,
        workspace_corners=True,
    )

    if SEND_TO_ROBOT:
        send_to_robot(robot_commands)
    else:
        print("\n[ROBOT] SEND_TO_ROBOT=False -> Es wurde nichts an den Roboter gesendet.")

    log_duration("total run", t_total)
    print("[RUN] Fertig.")


if __name__ == "__main__":
    freeze_support()
    main()



# ======================
# File: tests\run_solver.py
# ======================

import os
import sys
import cv2
from multiprocessing import freeze_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.Puzzle.Puzzle import Puzzle


IMAGE_PATH = r"assets\puzzle_images\Image (3).jpg"

DEBUG_DIR = os.path.join(PROJECT_ROOT, "assets", "DEBUG_Puzzle_Solver")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_debug_images(puzzle, prefix="debug"):
    debug_images = puzzle.get_debug_images()

    for i, img in enumerate(debug_images):
        out = os.path.join(DEBUG_DIR, f"{prefix}_{i:02d}.png")
        ok = cv2.imwrite(out, img)
        print(f"[DEBUG] Saved {out}: {ok}")


def main():
    print("[SOLVER] Starte Solver ohne ArUco...")
    print(f"[SOLVER] Input: {IMAGE_PATH}")

    puzzle = Puzzle(IMAGE_PATH, log_fn=print)

    try:
        print("[SOLVER] Extrahiere Puzzleteile...")
        puzzle.extract_pieces()

        print(f"[SOLVER] Extrahierte Teile: {len(puzzle.pieces_)}")

        save_debug_images(puzzle, prefix="extract")

        if len(puzzle.pieces_) == 0:
            raise ValueError("Keine Puzzleteile erkannt")

        print("[SOLVER] Starte Solver...")
        puzzle.solve_puzzle()

        print("[SOLVER] Puzzle erfolgreich gelöst")

    except Exception as e:
        print(f"[SOLVER] Fehler: {e}")

    finally:
        save_debug_images(puzzle, prefix="final")


if __name__ == "__main__":
    freeze_support()
    main()

# ======================
# File: tests\test_vision_pipeline.py
# ======================

import os
import sys
import cv2
import json
from multiprocessing import freeze_support

# =========================
# CONFIG
# =========================

IMAGE_PATH = r"assets\Bilder aruco marker\test\Image (2).jpg"

WORKSPACE_SIZE_PX = (1200, 800)
WORKSPACE_SIZE_MM = (400.0, 300.0)

CROP_MARGIN_RATIO_X = 0.12
CROP_MARGIN_RATIO_Y = 0.12

SAFE_Z_MM = 00.0
PICK_Z_MM = 00.0

# Erst auf False lassen, damit nur JSON ausgegeben wird.
SEND_TO_ROBOT = False
ROBOT_PORT = "COM3"

DEBUG_DIR = os.path.join("assets", "DEBUG")
os.makedirs(DEBUG_DIR, exist_ok=True)

ROBOT_ORIGIN_OFFSET_X_MM = 0.0
ROBOT_ORIGIN_OFFSET_Y_MM = 0.0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.Vision import VisionPipeline
from solver.Puzzle.Puzzle import Puzzle
from solver.Vision.robot_coordinates import RobotCoordinateMapper

try:
    from solver.Robot.robot_interface import RobotInterface
except Exception:
    RobotInterface = None


def show_resized(title, img, scale=0.6):
    preview = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(title, preview)


def parse_transform_report(log_line):
    parts = log_line.split()

    if len(parts) != 9 or parts[0] != "TRANSFORM_REPORT":
        return None

    return {
        "piece_id": int(parts[1]),
        "x0": float(parts[2]),
        "y0": float(parts[3]),
        "x1": float(parts[4]),
        "y1": float(parts[5]),
        "rotation_deg": float(parts[8]),
    }


def build_pick_place_sequence(cmd):
    """
    Erzeugt die Bewegungssequenz für ein einzelnes Puzzleteil.
    Werte bleiben intern in mm.
    robot_interface.py wandelt bei send_units='cm' automatisch in cm um.
    """
    return [
        {
            "type": "move",
            "description": "above_pick",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["pick_x_mm"],
            "y_mm": cmd["pick_y_mm"],
            "z_mm": SAFE_Z_MM,
        },
        {
            "type": "move",
            "description": "down_to_pick",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["pick_x_mm"],
            "y_mm": cmd["pick_y_mm"],
            "z_mm": PICK_Z_MM,
        },
        {
            "type": "suction",
            "description": "suction_on",
            "piece_id": cmd["piece_id"],
        },
        {
            "type": "move",
            "description": "lift_piece",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["pick_x_mm"],
            "y_mm": cmd["pick_y_mm"],
            "z_mm": SAFE_Z_MM,
        },
        {
            "type": "move",
            "description": "above_place",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["place_x_mm"],
            "y_mm": cmd["place_y_mm"],
            "z_mm": SAFE_Z_MM,
            "rotation_deg": cmd["rotation_deg"],
        },
        {
            "type": "move",
            "description": "down_to_place",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["place_x_mm"],
            "y_mm": cmd["place_y_mm"],
            "z_mm": PICK_Z_MM,
            "rotation_deg": cmd["rotation_deg"],
        },
        {
            "type": "suction",
            "description": "suction_off",
            "piece_id": cmd["piece_id"],
        },
        {
            "type": "move",
            "description": "lift_after_place",
            "piece_id": cmd["piece_id"],
            "x_mm": cmd["place_x_mm"],
            "y_mm": cmd["place_y_mm"],
            "z_mm": SAFE_Z_MM,
        },
    ]


def firmware_move_json(step):
    """
    Nur zur Anzeige: so sieht der MOVE-Befehl für die Firmware aus.
    Da die Firmware mit Steps pro mm rechnet, geben wir hier mm aus.
    """
    return {
        "MOVE": {
            "X": round((step["x_mm"] + ROBOT_ORIGIN_OFFSET_X_MM) / 10.0, 3),
            "Y": round((step["y_mm"] + ROBOT_ORIGIN_OFFSET_Y_MM) / 10.0, 3),
            "Z": round(step["z_mm"] / 10.0, 3),
        }
    }


def print_robot_plan(robot_commands):
    print("\n[ROBOT COMMANDS MM]")

    for cmd in robot_commands:
        print(
            f"Piece {cmd['piece_id']}: "
            f"Pick=({cmd['pick_x_mm']}, {cmd['pick_y_mm']}) mm, "
            f"Place=({cmd['place_x_mm']}, {cmd['place_y_mm']}) mm, "
            f"Rot={cmd['rotation_deg']}°"
        )

    print("\n[ROBOT JSON SEQUENCE FOR FIRMWARE]")

    for cmd in robot_commands:
        print(f"\nPiece {cmd['piece_id']}:")
        sequence = build_pick_place_sequence(cmd)

        for step in sequence:
            if step["type"] == "TODO":
                print(f"# TODO: {step['description']}")
                continue

            payload = firmware_move_json(step)
            print(json.dumps(payload))


def send_robot_plan(robot_commands):
    if RobotInterface is None:
        raise RuntimeError(
            "RobotInterface konnte nicht importiert werden. "
            "Prüfe solver/Robot/robot_interface.py und ob pyserial installiert ist."
        )

    robot = RobotInterface(port=ROBOT_PORT, send_units="cm")

    try:
        print("[ROBOT] READY")
        robot.ready()

        for cmd in robot_commands:
            print(f"[ROBOT] Starte Piece {cmd['piece_id']}")

            sequence = build_pick_place_sequence(cmd)

            for step in sequence:
                if step["type"] == "TODO":
                    print(f"[ROBOT TODO] {step['description']}")
                    continue

                robot.move_xyz_mm(
                step["x_mm"] + ROBOT_ORIGIN_OFFSET_X_MM,
                step["y_mm"] + ROBOT_ORIGIN_OFFSET_Y_MM,
                step["z_mm"],
                )

        print("[ROBOT] Fertig")
    finally:
        robot.close()


def main():
    pipeline = VisionPipeline(
        marker_length_mm=20.0,
        workspace_output_size_px=WORKSPACE_SIZE_PX,
        workspace_mm_size=WORKSPACE_SIZE_MM,
        aruco_ids=(0, 1, 2, 3),
    )

    result = pipeline.process_image_from_path(IMAGE_PATH)

    show_resized("ArUco Debug", result["aruco_debug"])
    show_resized("Warped Workspace", result["warped_workspace"])

    warped = result["warped_workspace"]
    if warped is None:
        raise ValueError("Warped workspace ist None -> ArUco Fehler")

    h, w = warped.shape[:2]
    margin_x = int(w * CROP_MARGIN_RATIO_X)
    margin_y = int(h * CROP_MARGIN_RATIO_Y)

    warped_inner = warped[margin_y:h - margin_y, margin_x:w - margin_x].copy()
    show_resized("Warped Inner", warped_inner)

    temp_dir = os.path.join(PROJECT_ROOT, "assets", "TEST")
    os.makedirs(temp_dir, exist_ok=True)

    warped_path = os.path.join(temp_dir, "warped_workspace_temp.png")
    cv2.imwrite(warped_path, warped_inner)

    cv2.imwrite(os.path.join(DEBUG_DIR, "warped_workspace.png"), warped)
    cv2.imwrite(os.path.join(DEBUG_DIR, "warped_inner.png"), warped_inner)

    print(f"[TEST] Warped workspace gespeichert unter: {warped_path}")
    print("[TEST] Starte Puzzle-Workflow...")

    transformation_logs = []

    def record_log(msg):
        print(msg)
        if msg.startswith("TRANSFORM_REPORT"):
            transformation_logs.append(msg)

    puzzle = Puzzle(warped_path, log_fn=record_log)

    puzzle.extract_pieces()

    debug_images = puzzle.get_debug_images()
    for i, img in enumerate(debug_images):
        path = os.path.join(DEBUG_DIR, f"piece_{i}.png")
        cv2.imwrite(path, img)
        print(f"[DEBUG] Saved: {path}")

    print(f"[TEST] Extrahierte Teile: {len(puzzle.pieces_)}")

    if len(puzzle.pieces_) == 0:
        raise ValueError("Keine Puzzleteile erkannt!")

    print("[TEST] Starte Solver...")
    puzzle.solve_puzzle()
    print("[TEST] Solver fertig")

    mapper = RobotCoordinateMapper(
        workspace_size_px=WORKSPACE_SIZE_PX,
        workspace_size_mm=WORKSPACE_SIZE_MM,
        crop_offset_px=(margin_x, margin_y),
    )

    robot_commands = []

    for log in transformation_logs:
        report = parse_transform_report(log)
        if report is None:
            continue

        cmd = mapper.transform_report_to_robot_command(report)
        robot_commands.append(cmd)

    robot_commands = sorted(robot_commands, key=lambda x: x["piece_id"])

    print_robot_plan(robot_commands)

    if SEND_TO_ROBOT:
        send_robot_plan(robot_commands)
    else:
        print("\n[ROBOT] SEND_TO_ROBOT=False -> Es wurde nichts an den Roboter gesendet.")

    debug_images = puzzle.get_debug_images()
    if debug_images:
        show_resized("Last Debug Image", debug_images[-1], scale=0.8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    freeze_support()
    main()