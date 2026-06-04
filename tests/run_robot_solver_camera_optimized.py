
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

USE_CAMERA = True
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

# ---------------------------------------------------------
# PICK-KALIBRATION
# ---------------------------------------------------------
# Wenn ein Teil gut und ein anderes schlecht gegriffen wird, reicht ein globaler
# PICK_OFFSET_X/Y nicht aus. Dann ist die Umrechnung Kamera/Workspace -> Roboter
# leicht verzerrt/skaliert/verdreht. Dafür kann hier eine affine Kalibrierung
# eingetragen werden.
#
# Vorgehen im Labor:
# 1) SEND_TO_ROBOT=False laufen lassen und die [PICK CAL] Zeilen notieren.
# 2) Roboter manuell exakt auf dieselben Schraubenloch-Zentren fahren.
# 3) Pro Punkt ein Paar eintragen:
#       ((detected_x_mm, detected_y_mm), (robot_x_mm, robot_y_mm))
#    detected_x/y kommen aus [PICK CAL] detected_workspace=(...)
#    robot_x/y sind die manuell abgelesenen echten Roboterkoordinaten.
# 4) Mindestens 3 Punkte, besser 4 Punkte über die Fläche verteilt verwenden.
#
# Beispiel:
# PICK_CALIBRATION_POINTS = [
#     ((74.6, 169.4), (275.9, 210.6)),
#     ((80.2,  64.7), (270.2, 315.3)),
#     ((237.2, 75.2), (113.3, 304.8)),
#     ((217.5,186.9), (133.0, 193.1)),
# ]
PICK_USE_AFFINE_CALIBRATION = True
PICK_CALIBRATION_POINTS = [
    ((230.17, 195.51), (127,192)),
    ((232.86,  77.10), (125,290)),
    (( 77.51, 187.48), (265,196)),
    (( 89.32,  75.20), (255,292)),
]

_PICK_AFFINE_CACHE = None

# Optionaler mechanischer Feinoffset nach der affine-Kalibrierung.
# Diesen nur verwenden, wenn ALLE Punkte gleichmässig daneben liegen.
PICK_FINE_OFFSET_X_MM = 0.0
PICK_FINE_OFFSET_Y_MM = 0.0

# A5-Zielfläche direkt über ihre vier Ecken im Roboterkoordinatensystem.
# Reihenfolge im Uhrzeigersinn: oben links, unten links, unten rechts, oben rechts.
# Wichtig: Das sind ROBOTERKOORDINATEN, nicht ArUco-/Bildkoordinaten.
A5_TOP_LEFT_ROBOT = (222.0, 90.0)
A5_BOTTOM_LEFT_ROBOT = (222.0, -30.0)
A5_BOTTOM_RIGHT_ROBOT = (32.0, -30.0)
A5_TOP_RIGHT_ROBOT = (32.0, 90.0)

A5_CENTER_MARGIN_X_MM = 5.0
A5_CENTER_MARGIN_Y_MM = 5.0

# ---------------------------------------------------------
# PLACE-KORREKTUR / A5-FEINTUNING
# ---------------------------------------------------------
# Diese Werte betreffen NUR das Platzieren auf dem A5, nicht das Picking.
# Positive/negative Richtung ist hier bewusst in ROBOTERKOORDINATEN angegeben.
# Aktueller Laborbefund:
# - Platzierung soll 15 mm weiter in negative Roboter-X-Richtung
# - Platzierung soll 5 mm weiter in negative Roboter-Y-Richtung
# Wichtig: Das muss hier negativ sein; positive X verschiebt in die falsche Richtung.
PLACE_ROBOT_FINE_OFFSET_X_MM = 14.0
PLACE_ROBOT_FINE_OFFSET_Y_MM = -2.0

# Firmware akzeptiert laut Fehlermeldung keine negativen Y-Werte.
# Deshalb wird die gesamte A5-Platzierung nach Gap + Fine-Offset automatisch
# wieder in den erlaubten Roboterbereich geschoben, ohne die relativen Abstände
# der Puzzleteile zu verändern.
PLACE_KEEP_WITHIN_ROBOT_BOUNDS = True
PLACE_ROBOT_MIN_SAFE_Y_MM = 2.0
PLACE_ROBOT_MAX_SAFE_Y_MM = 348.0
PLACE_ROBOT_MIN_SAFE_X_MM = 2.0
PLACE_ROBOT_MAX_SAFE_X_MM = 348.0

# Abstand zwischen den Puzzleteilen auf dem A5.
# Bei 2.0 mm werden die Teile ungefähr um je 1 mm vom Puzzlezentrum weg bewegt.
A5_INTER_PIECE_GAP_MM = 9.0

# Das A5 liegt im Roboter horizontal/landscape: lange Kante = X, kurze Kante = Y.
# Der Sauger-/Greifpunkt wird relativ zur linken oberen Puzzle-Kante platziert.
# Falls die Greifpunktrotation in der Praxis gespiegelt wirkt, diesen Wert auf -1.0 setzen.
GRIP_OFFSET_ROTATION_SIGN = 1.0

# C-Achse / Servo-Rotation:
# Der Roboter rotiert das Teil relativ zwischen Pick und Place.
# C_ROTATION_SIGN = 1.0 bedeutet: Solver-Winkel wird direkt als Servo-Delta verwendet.
# Falls die Teile exakt in die Gegenrichtung drehen, auf -1.0 setzen.
C_ROTATION_SIGN = 1.0
C_ROTATION_OFFSET_DEG = 0.0
C_SERVO_CENTER_DEG = 90.0
C_SERVO_MIN_DEG = 2.0
C_SERVO_MAX_DEG = 178.0

# Optionale manuelle Feinjustierung pro Puzzleteil.
# Erst nach dem nächsten SEND_TO_ROBOT=False Debug verwenden.
# Beispiel, falls Teil 3 in der Praxis 3° zu wenig im Uhrzeigersinn dreht:
# PIECE_ROTATION_FINE_OFFSETS_DEG = {3: 3.0}
PIECE_ROTATION_FINE_OFFSETS_DEG = {
    3: 0.0,
    4: 0.0,
}

# Offizielle Puzzle-/A5-Lage ist sehr streng und orthogonal.
# Darum werden Rotationen, die nur wenige Grad neben 0/90/180 liegen,
# auf den nächsten rechten Winkel geschnappt. Das verhindert die sichtbaren
# 3-5° Schräglagen in der realen Ablage.
A5_SNAP_ROTATIONS_TO_CARDINAL = True
# Bei diesem Wettbewerbspuzzle müssen die Teile orthogonal im A5 liegen.
# Deshalb wird nicht nur bei kleinen Abweichungen gesnappt, sondern praktisch
# jeder Winkel auf den nächsten 0/90/180/-90-Winkel gezogen.
A5_FORCE_CARDINAL_ROTATIONS = True
A5_CARDINAL_SNAP_TOLERANCE_DEG = 46.0


ROBOT_MIN_X_MM = 0.0
ROBOT_MAX_X_MM = 350.0
# Firmware akzeptiert gemäss NOTOK-Meldung nur Y >= 0.
# Darum darf die Softwarevalidierung negative Y-Werte nicht mehr erlauben.
ROBOT_MIN_Y_MM = 0.0
ROBOT_MAX_Y_MM = 350.0
ROBOT_MIN_Z_MM = 0.0
ROBOT_MAX_Z_MM = 18.0
# Damit das gelöste Puzzle nicht exakt auf der A5-Ecke beginnt,
# sondern etwas nach innen verschoben liegt.

SEND_TO_ROBOT = True
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


def get_pick_affine_matrix():
    """
    Berechnet eine affine Abbildung von Workspace-mm zu Roboter-mm:
        robot_x = a*x + b*y + c
        robot_y = d*x + e*y + f

    Dadurch werden nicht nur ein Offset, sondern auch Skalierung, leichte
    Verdrehung und Scherung korrigiert. Genau das braucht man, wenn ein Pickpunkt
    stimmt, ein anderer aber deutlich daneben liegt.
    """
    global _PICK_AFFINE_CACHE

    if _PICK_AFFINE_CACHE is not None:
        return _PICK_AFFINE_CACHE

    if not PICK_USE_AFFINE_CALIBRATION or len(PICK_CALIBRATION_POINTS) < 3:
        _PICK_AFFINE_CACHE = None
        return None

    src = np.asarray([p[0] for p in PICK_CALIBRATION_POINTS], dtype=np.float64)
    dst = np.asarray([p[1] for p in PICK_CALIBRATION_POINTS], dtype=np.float64)

    A = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])

    # Least Squares: funktioniert mit genau 3 Punkten und wird mit 4+ Punkten robuster.
    coeff_x, *_ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(A, dst[:, 1], rcond=None)

    M = np.vstack([coeff_x, coeff_y])
    _PICK_AFFINE_CACHE = M

    pred = A @ M.T
    err = pred - dst
    rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    print("[PICK CAL] affine workspace->robot aktiv")
    print(
        f"[PICK CAL] robot_x = {M[0,0]:.6f}*x + {M[0,1]:.6f}*y + {M[0,2]:.3f}"
    )
    print(
        f"[PICK CAL] robot_y = {M[1,0]:.6f}*x + {M[1,1]:.6f}*y + {M[1,2]:.3f}"
    )
    print(f"[PICK CAL] calibration RMS error = {rms:.3f} mm")

    for i, ((sx, sy), (rx, ry)) in enumerate(PICK_CALIBRATION_POINTS, start=1):
        px, py = pred[i - 1]
        print(
            f"[PICK CAL] p{i}: src=({sx:.2f},{sy:.2f}) "
            f"target=({rx:.2f},{ry:.2f}) pred=({px:.2f},{py:.2f}) "
            f"err=({px-rx:+.2f},{py-ry:+.2f})"
        )

    return M


def pick_to_robot(x_mm, y_mm):
    M = get_pick_affine_matrix()

    if M is not None:
        robot_x = M[0, 0] * x_mm + M[0, 1] * y_mm + M[0, 2]
        robot_y = M[1, 0] * x_mm + M[1, 1] * y_mm + M[1, 2]
    else:
        robot_x = PICK_OFFSET_X_MM + PICK_SIGN_X * x_mm
        robot_y = PICK_OFFSET_Y_MM + PICK_SIGN_Y * y_mm

    return (
        robot_x + PICK_FINE_OFFSET_X_MM,
        robot_y + PICK_FINE_OFFSET_Y_MM,
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

    # Feinkorrektur in echten Roboterkoordinaten.
    # Wichtig: nur Place, nicht Pick.
    robot_x += PLACE_ROBOT_FINE_OFFSET_X_MM
    robot_y += PLACE_ROBOT_FINE_OFFSET_Y_MM

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

def snap_rotation_to_cardinal_deg(angle_deg):
    """
    Snapt Rotationen auf exakt rechte Winkel.

    Früher wurde nur innerhalb einer kleinen Toleranz geschnappt. Das war für
    die A5-Ablage zu schwach: Winkel wie -16° oder -26° blieben sichtbar schräg.
    Im Wettbewerbsaufbau müssen die Teile aber orthogonal liegen, deshalb kann
    A5_FORCE_CARDINAL_ROTATIONS jeden Winkel auf den nächsten 0/90/180/-90-Winkel
    ziehen.

    Rückgabe: (gesnappter_winkel, angewendetes_delta).
    """
    angle = normalize_rotation_deg(angle_deg)

    if not A5_SNAP_ROTATIONS_TO_CARDINAL:
        return angle, 0.0

    candidates = [-180.0, -90.0, 0.0, 90.0, 180.0]
    best = min(candidates, key=lambda c: abs(normalize_rotation_deg(angle - c)))
    delta = normalize_rotation_deg(best - angle)

    if A5_FORCE_CARDINAL_ROTATIONS or abs(delta) <= A5_CARDINAL_SNAP_TOLERANCE_DEG:
        snapped = normalize_rotation_deg(best)
        return snapped, delta

    return angle, 0.0


def rotate_point_mm_around(point_xy, origin_xy, angle_deg):
    """Rotiert einen A5-mm-Punkt um einen A5-mm-Ursprung."""
    x, y = point_xy
    ox, oy = origin_xy
    rx, ry = rotate_point_px(x - ox, y - oy, angle_deg)
    return ox + rx, oy + ry


def piece_pixel_to_final_a5_mm(piece_id, col_px, row_px, align_info, cmd_by_id):
    """
    Wandelt einen gelösten Piece-Pixel in die tatsächlich geplante A5-Lage um.

    Das ist genauer als die alte Debug-Zeichnung, weil hier auch berücksichtigt wird:
    - Gap-Translation pro Teil
    - Safety-Shift
    - Rotation-Snap-Korrektur um den finalen Greifpunkt

    Dadurch zeigt das Debugbild nicht mehr nur die Solver-Rohlage, sondern die
    reale Ablage, die der Roboter ausführt.
    """
    base_x, base_y = solution_point_to_a5_mm(col_px, row_px, align_info)
    cmd = cmd_by_id.get(int(piece_id))

    if cmd is None:
        return base_x, base_y

    grip_base_x, grip_base_y = solution_point_to_a5_mm(
        cmd["raw_place_grip_px"][0],
        cmd["raw_place_grip_px"][1],
        align_info,
    )

    # Der echte Placepunkt wurde nachträglich durch Gap/Safety verschoben.
    dx = cmd["place_x_mm"] - grip_base_x
    dy = cmd["place_y_mm"] - grip_base_y

    x = base_x + dx
    y = base_y + dy

    snap_delta = float(cmd.get("rotation_snap_delta_deg", 0.0))
    if abs(snap_delta) > 1e-9:
        x, y = rotate_point_mm_around(
            (x, y),
            (cmd["place_x_mm"], cmd["place_y_mm"]),
            snap_delta,
        )

    return x, y


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




def estimate_piece_absolute_rotation_deg(piece_points_px, layout_rot_deg=0.0):
    """
    Schätzt die absolute Orientierung eines einzelnen gelösten Puzzleteils nach
    Anwendung der globalen Layout-Rotation.

    Wichtig: Für die A5-Ablage wollen wir nicht die *relative* Roboterdrehung
    snappen, sondern die *absolute* Zielorientierung des gelösten Teils.
    Sonst können Teile, die in der Quelle schräg liegen, fälschlich auf 0°
    Roboterdrehung kollabieren. Genau das ist bei Teil 3/4 passiert.
    """
    # piece_points_px kann eine Liste ODER ein NumPy-Array sein.
    # Deshalb hier nicht "if not piece_points_px" verwenden.
    if piece_points_px is None or len(piece_points_px) == 0:
        return 0.0

    pts = np.asarray([rotate_point_px(float(x), float(y), layout_rot_deg) for x, y in piece_points_px], dtype=np.float32)
    if pts.shape[0] < 5:
        return 0.0

    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle = rect

    long_side_angle = float(angle)
    if w < h:
        long_side_angle += 90.0

    while long_side_angle <= -90.0:
        long_side_angle += 180.0
    while long_side_angle > 90.0:
        long_side_angle -= 180.0

    return normalize_rotation_deg(long_side_angle)


def snap_absolute_piece_rotation_to_cardinal_deg(abs_angle_deg):
    """
    Wählt für eine absolute Teilorientierung den nächsten rechten Winkel und
    liefert die nötige Zusatzkorrektur zurück.

    Rückgabe: (target_abs_angle_deg, correction_delta_deg)
    """
    candidates = [-180.0, -90.0, 0.0, 90.0, 180.0]
    angle = normalize_rotation_deg(abs_angle_deg)
    target = min(candidates, key=lambda c: abs(normalize_rotation_deg(angle - c)))
    delta = normalize_rotation_deg(target - angle)
    return normalize_rotation_deg(target), delta

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

    # Zuerst die tatsächlich geplanten Puzzleformen zeichnen
    # inkl. Gap, Safety-Shift und Rotation-Snap.
    for idx, piece in enumerate(sorted(pieces, key=lambda p: p.id)):
        pts_mm = []
        for row, col in piece.pixels.keys():
            x_mm, y_mm = piece_pixel_to_final_a5_mm(
                int(piece.id),
                float(col),
                float(row),
                align_info,
                cmd_by_id,
            )
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
            f"P{cmd['piece_id']} A5=({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f}) / {cmd['rotation_deg']:.1f}deg [{cmd.get('place_point_source','?')}]",
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

    cmd_by_id = {int(cmd["piece_id"]): cmd for cmd in robot_commands}

    # Puzzleformen in der tatsächlich geplanten Ablage
    # inkl. Gap, Safety-Shift und Rotation-Snap.
    for idx, piece in enumerate(sorted(pieces, key=lambda p: p.id)):
        pts_mm = []
        for row, col in piece.pixels.keys():
            x_mm, y_mm = piece_pixel_to_final_a5_mm(
                int(piece.id),
                float(col),
                float(row),
                align_info,
                cmd_by_id,
            )
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

        robot_x, robot_y = place_to_robot(cmd["place_x_mm"], cmd["place_y_mm"])
        label_a5 = f"P{cmd['piece_id']} A5=({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f})"
        label_robot = f"Robot=({robot_x:.1f}, {robot_y:.1f}) rot={cmd['rotation_deg']:.1f}deg"
        cv2.putText(debug, label_a5, (fx + 10, fy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(debug, label_robot, (fx + 10, fy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,180), 1, cv2.LINE_AA)

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



def snapshot_piece_points_by_id(pieces):
    """
    Speichert die ursprünglichen Pixelpunkte vor puzzle.solve_puzzle().
    Wichtig: Puzzle.solve_puzzle() mutiert piece.pixels, darum müssen wir vor dem
    Solver eine Kopie behalten, wenn wir die echte Drehung später vergleichen wollen.
    Rückgabe: piece_id -> ndarray mit Punkten (x=col, y=row).
    """
    snapshots = {}
    for piece in pieces:
        pts = []
        for row, col in piece.pixels.keys():
            pts.append((float(col), float(row)))
        if pts:
            snapshots[int(piece.id)] = np.asarray(pts, dtype=np.float32)
    return snapshots


def _downsample_points(points, max_points=2500):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= max_points:
        return pts
    step = max(1, int(math.ceil(pts.shape[0] / max_points)))
    return pts[::step]


def _rotation_match_score(src_rel_xy, target_distance, origin_xy, angle_deg):
    """
    Bewertet, wie gut die Quellform nach Rotation auf die Zielform passt.
    Kleine Werte sind besser.
    """
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)

    x = src_rel_xy[:, 0]
    y = src_rel_xy[:, 1]

    rx = x * ca - y * sa + origin_xy[0]
    ry = x * sa + y * ca + origin_xy[1]

    ix = np.round(rx).astype(np.int32)
    iy = np.round(ry).astype(np.int32)

    valid = (
        (ix >= 0)
        & (iy >= 0)
        & (ix < target_distance.shape[1])
        & (iy < target_distance.shape[0])
    )

    if np.count_nonzero(valid) < max(20, int(0.5 * len(src_rel_xy))):
        return float("inf")

    return float(np.mean(target_distance[iy[valid], ix[valid]]))


def estimate_piece_rotation_from_shape(initial_points_xy, initial_pick_xy, solved_points_xy, solved_pick_xy, piece_id=None):
    """
    Schätzt die ECHTE benötigte Rotation eines Teils über Form-Matching.

    Warum: TRANSFORM_REPORT liefert bei manchen Teilen nur eine lokale/teilweise
    Rotation aus dem Solver-Schritt. Bei den letzten Teilen kann dieser Winkel
    trotz guter Place-Koordinate falsch sein. Diese Funktion vergleicht stattdessen
    die ursprüngliche Teilform mit der final gelösten Teilform, beide relativ zum
    Greifpunkt/Schraubenloch.
    """
    if initial_points_xy is None or solved_points_xy is None:
        return None

    src = np.asarray(initial_points_xy, dtype=np.float32) - np.asarray(initial_pick_xy, dtype=np.float32)
    tgt = np.asarray(solved_points_xy, dtype=np.float32) - np.asarray(solved_pick_xy, dtype=np.float32)

    if src.shape[0] < 50 or tgt.shape[0] < 50:
        return None

    src_s = _downsample_points(src, max_points=2500)
    tgt_s = _downsample_points(tgt, max_points=6000)

    all_pts = np.vstack([src_s, tgt_s])
    min_x = float(np.min(all_pts[:, 0]))
    min_y = float(np.min(all_pts[:, 1]))
    max_x = float(np.max(all_pts[:, 0]))
    max_y = float(np.max(all_pts[:, 1]))

    pad = 35
    width = int(math.ceil(max_x - min_x)) + 2 * pad + 1
    height = int(math.ceil(max_y - min_y)) + 2 * pad + 1

    if width <= 5 or height <= 5 or width > 2000 or height > 2000:
        return None

    origin = np.asarray([pad - min_x, pad - min_y], dtype=np.float32)

    # Distanzbild der Zielkontur/-fläche: Zielpunkte sind 0, Umgebung hat Abstand.
    target_img = np.full((height, width), 255, dtype=np.uint8)
    tx = np.round(tgt_s[:, 0] + origin[0]).astype(np.int32)
    ty = np.round(tgt_s[:, 1] + origin[1]).astype(np.int32)
    valid_t = (tx >= 0) & (ty >= 0) & (tx < width) & (ty < height)
    target_img[ty[valid_t], tx[valid_t]] = 0

    # Leicht dilatieren, damit Pixel-Rasterung nicht zu hart bestraft.
    target_zero = (target_img == 0).astype(np.uint8) * 255
    target_zero = cv2.dilate(target_zero, np.ones((3, 3), np.uint8), iterations=1)
    target_img = np.full((height, width), 255, dtype=np.uint8)
    target_img[target_zero > 0] = 0

    dist = cv2.distanceTransform(target_img, cv2.DIST_L2, 3)

    best_angle = 0.0
    best_score = float("inf")

    # Grobsuche über alle Winkel.
    for angle in np.arange(-180.0, 180.0, 2.0):
        score = _rotation_match_score(src_s, dist, origin, angle)
        if score < best_score:
            best_score = score
            best_angle = float(angle)

    # Feinsuche rund um den besten Winkel.
    fine_start = best_angle - 3.0
    fine_end = best_angle + 3.001
    for angle in np.arange(fine_start, fine_end, 0.25):
        norm_angle = normalize_rotation_deg(float(angle))
        score = _rotation_match_score(src_s, dist, origin, norm_angle)
        if score < best_score:
            best_score = score
            best_angle = norm_angle

    best_angle = normalize_rotation_deg(best_angle)

    if piece_id is not None:
        print(
            f"[ROT MATCH] Piece {piece_id}: shape_rotation={best_angle:.2f}°, "
            f"score={best_score:.2f}"
        )

    return best_angle

def _candidate_is_valid_hole_center(rr, cc, inner_mask, dist):
    if not (0 <= rr < inner_mask.shape[0] and 0 <= cc < inner_mask.shape[1]):
        return False
    if inner_mask[rr, cc] == 0:
        return False
    if float(dist[rr, cc]) < 8.0:
        return False
    return True


def _circle_fit_least_squares(points_xy):
    """
    Least-Squares-Kreisfit für Punkte im Format (x, y).
    Gibt (cx, cy, r, residual) zurück oder None.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 8:
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None

    r = float(math.sqrt(r2))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = float(np.mean(np.abs(d - r)))

    return float(cx), float(cy), r, residual


def _refine_hole_center_with_edges(gray, inner_mask, dist, coarse_row, coarse_col, patch_radius=20):
    """
    Verfeinert einen groben Schraubenloch-Kandidaten über lokale Kreis-/Kanteninformation.
    Ziel: nicht den hellsten Reflex greifen, sondern das Zentrum des runden Loch-/Ringbereichs.
    """
    h, w = gray.shape[:2]
    cr = int(round(coarse_row))
    cc = int(round(coarse_col))

    y0 = max(0, cr - patch_radius)
    y1 = min(h, cr + patch_radius + 1)
    x0 = max(0, cc - patch_radius)
    x1 = min(w, cc + patch_radius + 1)

    if y1 <= y0 or x1 <= x0:
        return coarse_row, coarse_col, 0.0

    patch = gray[y0:y1, x0:x1]
    patch_mask = inner_mask[y0:y1, x0:x1]

    if patch.size == 0 or np.count_nonzero(patch_mask) < 20:
        return coarse_row, coarse_col, 0.0

    patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)

    # CLAHE macht den Metallring/Lochrand kontrastreicher.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    patch_eq = clahe.apply(patch_blur)

    # Kanten im lokalen Patch suchen.
    edges = cv2.Canny(patch_eq, 35, 100)
    edges = cv2.bitwise_and(edges, edges, mask=patch_mask)

    ys, xs = np.where(edges > 0)

    if len(xs) < 8:
        return coarse_row, coarse_col, 0.0

    # Nur Kanten im sinnvollen Radius um den groben Kandidaten verwenden.
    local_cx = cc - x0
    local_cy = cr - y0
    d = np.sqrt((xs - local_cx) ** 2 + (ys - local_cy) ** 2)
    keep = (d >= 3.0) & (d <= 18.0)

    xs = xs[keep]
    ys = ys[keep]

    if len(xs) < 8:
        return coarse_row, coarse_col, 0.0

    fit = _circle_fit_least_squares(np.column_stack([xs, ys]))

    if fit is None:
        return coarse_row, coarse_col, 0.0

    cx, cy, r, residual = fit

    # Schraubenloch/Ring ist klein. Ausreisser ignorieren.
    if r < 3.0 or r > 16.0:
        return coarse_row, coarse_col, 0.0

    if residual > 3.5:
        return coarse_row, coarse_col, 0.0

    refined_row = y0 + cy
    refined_col = x0 + cx

    rr = int(round(refined_row))
    cc2 = int(round(refined_col))

    if not _candidate_is_valid_hole_center(rr, cc2, inner_mask, dist):
        return coarse_row, coarse_col, 0.0

    confidence = max(0.0, 1.0 - residual / 3.5)
    return float(refined_row), float(refined_col), confidence


def detect_screw_hole_center(piece, source_img, debug_img=None):
    """
    Erkennt das Schraubenloch eines Puzzleteils genauer.

    Verbesserungen gegenüber der alten Version:
    - HoughCircles wird immer genutzt, nicht nur als Fallback.
    - helle Konturen werden weiterhin als Kandidaten genutzt.
    - jeder Kandidat wird lokal mit Canny-Kanten + Kreisfit verfeinert.
    - dadurch landet der Pickpunkt eher im Ringzentrum statt auf einem hellen Reflex.
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

    # Randbereich entfernen, damit Einbuchtungen/Kanten nicht als Loch erkannt werden.
    erode_kernel = np.ones((13, 13), np.uint8)
    inner_mask = cv2.erode(piece_mask, erode_kernel, iterations=1)

    if np.count_nonzero(inner_mask) < 50:
        inner_mask = piece_mask.copy()

    dist = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # ------------------------------------------------------------------
    # Methode A: HoughCircles direkt auf lokaler Kontrastverbesserung
    # ------------------------------------------------------------------
    masked_gray = cv2.bitwise_and(gray_blur, gray_blur, mask=inner_mask)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(masked_gray)

    circles = cv2.HoughCircles(
        eq,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=70,
        param2=8,
        minRadius=3,
        maxRadius=13,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for x, y, r in circles:
            if not _candidate_is_valid_hole_center(int(y), int(x), inner_mask, dist):
                continue

            local = gray[
                max(0, y - 8):min(gray.shape[0], y + 9),
                max(0, x - 8):min(gray.shape[1], x + 9),
            ]

            local_brightness = float(np.mean(local)) if local.size else 0.0
            edge_distance = float(dist[int(y), int(x)])

            candidates.append({
                "row": float(min_row + y),
                "col": float(min_col + x),
                "row_roi": float(y),
                "col_roi": float(x),
                "radius": float(r),
                "area": float(np.pi * r * r),
                "circularity": 1.0,
                "edge_distance": edge_distance,
                "score": 350.0 + edge_distance * 4.0 + local_brightness * 0.15 - abs(r - 6.0) * 3.0,
                "method": "hough",
            })

    # ------------------------------------------------------------------
    # Methode B: helle Metall-/Ringbereiche als Konturen
    # ------------------------------------------------------------------
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

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 8 or area > 450:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.20:
            continue

        # Momenten-Zentrum ist bei Teilreflexen oft besser als minEnclosingCircle.
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
        else:
            (x, y), _ = cv2.minEnclosingCircle(cnt)

        (_, _), radius = cv2.minEnclosingCircle(cnt)

        if radius < 2.0 or radius > 15.0:
            continue

        rr = int(round(y))
        cc = int(round(x))

        if not _candidate_is_valid_hole_center(rr, cc, inner_mask, dist):
            continue

        local = gray[
            max(0, rr - 5):min(gray.shape[0], rr + 6),
            max(0, cc - 5):min(gray.shape[1], cc + 6),
        ]

        local_brightness = float(np.mean(local)) if local.size else 0.0
        edge_distance = float(dist[rr, cc])

        score = (
            circularity * 120.0
            + edge_distance * 4.0
            + local_brightness * 0.25
            - abs(radius - 5.0) * 3.0
        )

        candidates.append({
            "row": float(min_row + y),
            "col": float(min_col + x),
            "row_roi": float(y),
            "col_roi": float(x),
            "radius": float(radius),
            "area": float(area),
            "circularity": float(circularity),
            "edge_distance": edge_distance,
            "score": score,
            "method": "bright_contour",
        })

    # ------------------------------------------------------------------
    # Kandidaten lokal über Ring-/Kanten-Kreisfit verfeinern
    # ------------------------------------------------------------------
    refined_candidates = []

    for cand in candidates:
        coarse_row_roi = cand["row_roi"]
        coarse_col_roi = cand["col_roi"]

        refined_row_roi, refined_col_roi, refine_conf = _refine_hole_center_with_edges(
            gray,
            inner_mask,
            dist,
            coarse_row_roi,
            coarse_col_roi,
        )

        # Kandidat nicht zu weit verschieben, sonst war der Kreisfit auf einer Fremdkante.
        shift = math.hypot(refined_col_roi - coarse_col_roi, refined_row_roi - coarse_row_roi)
        if shift > 9.0:
            refined_row_roi = coarse_row_roi
            refined_col_roi = coarse_col_roi
            refine_conf = 0.0

        rr = int(round(refined_row_roi))
        cc = int(round(refined_col_roi))

        if not _candidate_is_valid_hole_center(rr, cc, inner_mask, dist):
            continue

        cand = dict(cand)
        cand["row"] = float(min_row + refined_row_roi)
        cand["col"] = float(min_col + refined_col_roi)
        cand["row_roi"] = float(refined_row_roi)
        cand["col_roi"] = float(refined_col_roi)
        cand["refine_conf"] = float(refine_conf)
        cand["score"] += refine_conf * 120.0 - shift * 2.0

        refined_candidates.append(cand)

    if refined_candidates:
        best = max(refined_candidates, key=lambda c: c["score"])

        row = float(best["row"])
        col = float(best["col"])

        if debug_img is not None:
            cv2.circle(debug_img, (int(round(col)), int(round(row))), 9, (0, 255, 255), -1)
            cv2.circle(debug_img, (int(round(col)), int(round(row))), 15, (255, 0, 255), 2)
            cv2.putText(
                debug_img,
                f"H{piece.id}",
                (int(round(col)) + 10, int(round(row)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_img,
                f"{best['method']} r={best['radius']:.1f}",
                (int(round(col)) + 10, int(round(row)) + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        print(
            f"[HOLE] Piece {piece.id}: "
            f"row={row:.1f}, col={col:.1f}, "
            f"method={best['method']}, "
            f"r={best['radius']:.1f}, "
            f"circ={best['circularity']:.2f}, "
            f"dist={best['edge_distance']:.1f}, "
            f"refine={best.get('refine_conf', 0.0):.2f}, "
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
    """
    Wandelt den gewünschten Puzzle-Rotationswinkel in zwei Servo-C-Winkel um.

    Wichtig: Für den Sauger zählt die relative Änderung zwischen Pick und Place:
        delta = final_rot - pre_rot

    Die alte Version setzte final_rot immer auf 0° oder 180°. Dadurch wurde bei
    kleinen positiven Winkeln wegen der Sicherheits-Clamps auf 2°/178° ein Teil
    der Rotation abgeschnitten. Beispiel: +4.6° wurde effektiv nur ca. +2.6°.

    Neue Logik: Delta symmetrisch um die Servo-Mitte 90° verteilen:
        pre_rot   = 90° - delta/2
        final_rot = 90° + delta/2

    Dadurch bleiben auch kleine Winkel exakt und grosse Winkel nutzen weiterhin
    fast den ganzen Servo-Bereich.
    """
    desired_delta = normalize_rotation_deg(
        C_ROTATION_SIGN * angle_deg + C_ROTATION_OFFSET_DEG
    )

    max_delta = C_SERVO_MAX_DEG - C_SERVO_MIN_DEG

    if desired_delta > max_delta:
        print(
            f"[ROT WARN] Gewünschte Rotation {desired_delta:.2f}° > "
            f"Servo-Max-Delta {max_delta:.2f}° -> wird begrenzt"
        )
        desired_delta = max_delta

    if desired_delta < -max_delta:
        print(
            f"[ROT WARN] Gewünschte Rotation {desired_delta:.2f}° < "
            f"-Servo-Max-Delta {-max_delta:.2f}° -> wird begrenzt"
        )
        desired_delta = -max_delta

    pre_rot = C_SERVO_CENTER_DEG - desired_delta / 2.0
    final_rot = C_SERVO_CENTER_DEG + desired_delta / 2.0

    pre_rot = max(C_SERVO_MIN_DEG, min(C_SERVO_MAX_DEG, pre_rot))
    final_rot = max(C_SERVO_MIN_DEG, min(C_SERVO_MAX_DEG, final_rot))

    return pre_rot, final_rot

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

    pre_rot = max(C_SERVO_MIN_DEG, min(C_SERVO_MAX_DEG, pre_rot))
    final_rot = max(C_SERVO_MIN_DEG, min(C_SERVO_MAX_DEG, final_rot))

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
        pick_robot_x, pick_robot_y = pick_to_robot(
            cmd["pick_x_mm"],
            cmd["pick_y_mm"],
        )
        place_robot_x, place_robot_y = place_to_robot(
            cmd["place_x_mm"],
            cmd["place_y_mm"],
        )

        print(
            f"\nPiece {cmd['piece_id']}: "
            f"PickAruco=({cmd['pick_x_mm']:.1f}, {cmd['pick_y_mm']:.1f}) mm | "
            f"PickRobot=({pick_robot_x:.1f}, {pick_robot_y:.1f}) mm | "
            f"PlaceA5=({cmd['place_x_mm']:.1f}, {cmd['place_y_mm']:.1f}) mm | "
            f"PlaceRobot=({place_robot_x:.1f}, {place_robot_y:.1f}) mm | "
            f"Gap={cmd.get('a5_gap_applied_mm', 0.0):.1f} mm "
            f"({cmd.get('a5_gap_dx_mm', 0.0):+.1f}, {cmd.get('a5_gap_dy_mm', 0.0):+.1f}) | "
            f"Rot={cmd['rotation_deg']:.1f}° "
            f"[{cmd.get('rotation_source', 'unknown')}, fine={cmd.get('rotation_fine_offset_deg', 0.0):+.1f}°]"
        )
        print(
            f"  [PICK CAL] Piece {cmd['piece_id']}: "
            f"detected_workspace=({cmd['pick_x_mm']:.2f}, {cmd['pick_y_mm']:.2f}) "
            f"current_robot_target=({pick_robot_x:.2f}, {pick_robot_y:.2f})"
        )
        if "rotation_from_transform_report_deg" in cmd:
            print(
                f"  [ROT DBG] transform_report={cmd['rotation_from_transform_report_deg']:.1f}° "
                f"-> shape_match={cmd['rotation_deg']:.1f}°"
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

def validate_robot_plan(robot_commands):
    """
    Prüft alle berechneten MOVE-Positionen, bevor der Roboter überhaupt startet.

    Wichtig: build_pick_place_sequence enthält auch rotate- und suction-Steps.
    Diese besitzen keine x_mm/y_mm/z_mm und dürfen deshalb nicht wie MOVE-Steps
    validiert oder als Firmware-MOVE ausgegeben werden.
    """
    print("\n[ROBOT CHECK] Prüfe alle MOVE-Positionen vor dem Senden...")

    for cmd in robot_commands:
        for step in build_pick_place_sequence(cmd):
            if step.get("type") != "move":
                continue

            validate_robot_position(
                step["x_mm"],
                step["y_mm"],
                step["z_mm"],
                step.get("rotation_deg", 90.0),
                f"Piece {cmd['piece_id']} / {step['description']}",
            )

    print("[ROBOT CHECK] Alle MOVE-Positionen sind innerhalb der erlaubten Grenzen.")


def validate_robot_position(x, y, z, c, description=""):
    errors = []

    if not (ROBOT_MIN_X_MM <= x <= ROBOT_MAX_X_MM):
        errors.append(f"X={x:.2f} ausserhalb [{ROBOT_MIN_X_MM}, {ROBOT_MAX_X_MM}]")

    if not (ROBOT_MIN_Y_MM <= y <= ROBOT_MAX_Y_MM):
        errors.append(f"Y={y:.2f} ausserhalb [{ROBOT_MIN_Y_MM}, {ROBOT_MAX_Y_MM}]")

    if not (ROBOT_MIN_Z_MM <= z <= ROBOT_MAX_Z_MM):
        errors.append(f"Z={z:.2f} ausserhalb [{ROBOT_MIN_Z_MM}, {ROBOT_MAX_Z_MM}]")

    if not (C_SERVO_MIN_DEG <= c <= C_SERVO_MAX_DEG):
        errors.append(
            f"C={c:.2f} ausserhalb [{C_SERVO_MIN_DEG}, {C_SERVO_MAX_DEG}]"
        )

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

        if c_deg < C_SERVO_MIN_DEG:
            return int(round(C_SERVO_MIN_DEG))

        if c_deg > C_SERVO_MAX_DEG:
            return int(round(C_SERVO_MAX_DEG))

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


def apply_inter_piece_gap(robot_commands, align_info):
    """
    Fügt einen kleinen Abstand zwischen den Puzzleteilen hinzu.

    Wichtig:
    Die alte Variante hat nur geprüft, ob ein Teil links/rechts bzw. oben/unten
    vom Gesamtzentrum liegt. Bei asymmetrischen Teilen kann das die falsche
    Hälfte treffen oder zu wenig Abstand erzeugen.

    Diese Version bestimmt zuerst die relative 2x2-Position der gelösten Teile
    über ihre BBox-Zentren:
        linke Spalte  -> -gap/2 in A5-X
        rechte Spalte -> +gap/2 in A5-X
        obere Zeile   -> -gap/2 in A5-Y
        untere Zeile  -> +gap/2 in A5-Y

    Dadurch bleibt die Puzzle-Logik stabiler und die Teile werden gleichmässig
    auseinandergezogen.
    """
    gap = float(A5_INTER_PIECE_GAP_MM)

    if gap <= 0.0 or not robot_commands:
        return

    refs = []

    for cmd in robot_commands:
        ref_px = cmd.get("solved_bbox_center_px", cmd.get("raw_place_grip_px"))
        if ref_px is None:
            continue

        ref_x_mm, ref_y_mm = solution_point_to_a5_mm(
            ref_px[0],
            ref_px[1],
            align_info,
        )

        refs.append((cmd, ref_x_mm, ref_y_mm))

    if not refs:
        return

    xs = sorted(r[1] for r in refs)
    ys = sorted(r[2] for r in refs)

    split_x = (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2.0 if len(xs) >= 2 else xs[0]
    split_y = (ys[len(ys) // 2 - 1] + ys[len(ys) // 2]) / 2.0 if len(ys) >= 2 else ys[0]

    half_gap = gap / 2.0

    for cmd, ref_x_mm, ref_y_mm in refs:
        dx = -half_gap if ref_x_mm < split_x else half_gap
        dy = -half_gap if ref_y_mm < split_y else half_gap

        cmd["place_x_mm"] += dx
        cmd["place_y_mm"] += dy

        cmd["a5_gap_applied_mm"] = gap
        cmd["a5_gap_dx_mm"] = dx
        cmd["a5_gap_dy_mm"] = dy

    print(
        f"[A5 GAP] grid-basiert angewendet: gap={gap:.1f} mm, "
        f"split=({split_x:.1f}, {split_y:.1f})"
    )


def apply_grid_overlap_separation(robot_commands, min_gap_mm=3.0):
    """
    Zusätzliche robuste Entzerrung der 2x2-Ablage.

    Die normale Gap-Funktion verschiebt die Punkte vom Zentrum weg. Wenn die
    Solver-/Formlage aber leicht schief ist, kann es trotzdem zu Überlagerungen
    kommen. Diese Funktion arbeitet rein auf den roten Placepunkten und erzwingt
    einen Mindestabstand zwischen linker/rechter Spalte und oberer/unterer Zeile.

    Sie verschiebt wieder nur die A5-Placepunkte, nicht die Pickpunkte.
    """
    if len(robot_commands) < 4:
        return

    cmds = list(robot_commands)
    xs = sorted(c["place_x_mm"] for c in cmds)
    ys = sorted(c["place_y_mm"] for c in cmds)
    split_x = (xs[1] + xs[2]) / 2.0
    split_y = (ys[1] + ys[2]) / 2.0

    left = [c for c in cmds if c["place_x_mm"] < split_x]
    right = [c for c in cmds if c["place_x_mm"] >= split_x]
    top = [c for c in cmds if c["place_y_mm"] < split_y]
    bottom = [c for c in cmds if c["place_y_mm"] >= split_y]

    if not left or not right or not top or not bottom:
        return

    # Abstand zwischen den Loch-/Greifpunkten ist nicht gleich Teilkantenabstand,
    # aber für euer 2x2-Puzzle ist dies ein stabiler zusätzlicher Separator.
    # Wir schieben beide Seiten symmetrisch auseinander.
    current_col_gap = min(c["place_x_mm"] for c in right) - max(c["place_x_mm"] for c in left)
    current_row_gap = min(c["place_y_mm"] for c in bottom) - max(c["place_y_mm"] for c in top)

    push_x = max(0.0, min_gap_mm - current_col_gap) / 2.0
    push_y = max(0.0, min_gap_mm - current_row_gap) / 2.0

    if push_x <= 1e-9 and push_y <= 1e-9:
        return

    for c in left:
        c["place_x_mm"] -= push_x
        c["a5_overlap_sep_dx_mm"] = c.get("a5_overlap_sep_dx_mm", 0.0) - push_x
    for c in right:
        c["place_x_mm"] += push_x
        c["a5_overlap_sep_dx_mm"] = c.get("a5_overlap_sep_dx_mm", 0.0) + push_x
    for c in top:
        c["place_y_mm"] -= push_y
        c["a5_overlap_sep_dy_mm"] = c.get("a5_overlap_sep_dy_mm", 0.0) - push_y
    for c in bottom:
        c["place_y_mm"] += push_y
        c["a5_overlap_sep_dy_mm"] = c.get("a5_overlap_sep_dy_mm", 0.0) + push_y

    print(
        f"[A5 OVERLAP SEP] push=({push_x:.2f}, {push_y:.2f}) mm, "
        f"place_point_gap_before=({current_col_gap:.2f}, {current_row_gap:.2f}) mm"
    )



def shift_a5_place_coordinates_by_robot_delta(robot_commands, dx_robot, dy_robot):
    """
    Verschiebt alle A5-Place-Koordinaten um einen Delta in echten
    Roboterkoordinaten. Dadurch bleiben die relativen Abstände zwischen den
    Puzzleteilen erhalten.
    """
    delta_x_a5 = dx_robot * A5_AXIS_X_UNIT[0] + dy_robot * A5_AXIS_X_UNIT[1]
    delta_y_a5 = dx_robot * A5_AXIS_Y_UNIT[0] + dy_robot * A5_AXIS_Y_UNIT[1]

    for cmd in robot_commands:
        cmd["place_x_mm"] += delta_x_a5
        cmd["place_y_mm"] += delta_y_a5

    return delta_x_a5, delta_y_a5


def keep_place_points_inside_robot_bounds(robot_commands):
    """
    Stellt sicher, dass die Place-Positionen nach Fine-Offset und Gap noch im
    von der Firmware akzeptierten Roboterbereich liegen.

    Wichtig: Es wird NICHT pro Teil geclamped, weil das die Puzzle-Geometrie
    verzerren würde. Stattdessen wird die komplette Lösung als Ganzes verschoben.
    """
    if not PLACE_KEEP_WITHIN_ROBOT_BOUNDS or not robot_commands:
        return (0.0, 0.0)

    coords = [place_to_robot(cmd["place_x_mm"], cmd["place_y_mm"]) for cmd in robot_commands]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    dx = 0.0
    dy = 0.0

    if min(xs) < PLACE_ROBOT_MIN_SAFE_X_MM:
        dx = PLACE_ROBOT_MIN_SAFE_X_MM - min(xs)
    elif max(xs) > PLACE_ROBOT_MAX_SAFE_X_MM:
        dx = PLACE_ROBOT_MAX_SAFE_X_MM - max(xs)

    if min(ys) < PLACE_ROBOT_MIN_SAFE_Y_MM:
        dy = PLACE_ROBOT_MIN_SAFE_Y_MM - min(ys)
    elif max(ys) > PLACE_ROBOT_MAX_SAFE_Y_MM:
        dy = PLACE_ROBOT_MAX_SAFE_Y_MM - max(ys)

    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
        da5x, da5y = shift_a5_place_coordinates_by_robot_delta(robot_commands, dx, dy)
        print(
            f"[A5 SAFETY SHIFT] Placepunkte global verschoben: "
            f"robot_delta=({dx:+.2f}, {dy:+.2f}) mm -> "
            f"a5_delta=({da5x:+.2f}, {da5y:+.2f}) mm"
        )

    return dx, dy


def deduplicate_robot_commands_by_piece(robot_commands):
    """
    Der Backtracking-Solver kann mehrere TRANSFORM_REPORT-Zeilen für dasselbe
    Piece ausgeben. Für den Roboter darf jedes physische Teil aber nur einmal
    gepickt und platziert werden. Wir behalten deshalb pro piece_id den letzten
    berechneten Command.
    """
    unique = {}
    duplicates = []

    for cmd in robot_commands:
        pid = int(cmd["piece_id"])
        if pid in unique:
            duplicates.append(pid)
        unique[pid] = cmd

    if duplicates:
        dup_text = ", ".join(str(pid) for pid in sorted(set(duplicates)))
        print(f"[WARN] Doppelte Robot-Commands entfernt für Piece(s): {dup_text}")

    return [unique[pid] for pid in sorted(unique.keys())]

def align_solution_to_a5(robot_commands, solution_points_px, solved_piece_points_map=None):
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

        # WICHTIG:
        # Die Roboterdrehung ist eine RELATIVE Drehung vom aktuellen Pick-Zustand
        # in die gewünschte A5-Zielorientierung.
        #
        # Frühere Versionen haben die relative Roboterdrehung direkt auf
        # 0/90/180 gesnappt. Das war falsch: Dadurch konnten schräg liegende
        # Ausgangsteile (insb. Teil 3/4) fälschlich auf 0° kollabieren und wurden
        # in der Praxis gar nicht oder zu wenig gedreht.
        #
        # Korrekte Strategie:
        # 1) shape_match liefert die relative Rohdrehung Quelle -> Solver-Lösung
        # 2) layout_rot dreht die gesamte Solver-Lösung horizontal ins A5
        # 3) pro Teil bestimmen wir die ABSOLUTE Zielorientierung nach layout_rot
        # 4) diese absolute Zielorientierung wird auf den nächsten rechten Winkel
        #    geschnappt; nur diese Zusatzkorrektur wird auf die Roboterdrehung addiert
        base_relative_rotation = normalize_rotation_deg(
            cmd["rotation_deg"] + best["layout_rot"]
        )

        piece_abs_before_deg = None
        piece_abs_target_deg = None
        piece_abs_cardinal_delta_deg = 0.0

        if solved_piece_points_map is not None:
            piece_pts = solved_piece_points_map.get(int(cmd["piece_id"]))

            # piece_pts kann ein NumPy-Array sein. Darum NICHT "if piece_pts:" verwenden,
            # weil das bei Arrays zu "truth value is ambiguous" führt.
            if piece_pts is not None and len(piece_pts) > 0:
                piece_abs_before_deg = estimate_piece_absolute_rotation_deg(
                    piece_pts,
                    best["layout_rot"],
                )
                piece_abs_target_deg, piece_abs_cardinal_delta_deg = snap_absolute_piece_rotation_to_cardinal_deg(
                    piece_abs_before_deg
                )

        piece_rot_fine = float(PIECE_ROTATION_FINE_OFFSETS_DEG.get(int(cmd["piece_id"]), 0.0))

        # Gesamte zusätzliche Zielkorrektur in der A5-Ebene:
        # absolute Orthogonalisierung + optionale manuelle Feinjustierung.
        total_visual_delta_deg = normalize_rotation_deg(
            piece_abs_cardinal_delta_deg + piece_rot_fine
        )

        final_relative_rotation = normalize_rotation_deg(
            base_relative_rotation + total_visual_delta_deg
        )

        cmd["rotation_deg_raw_before_snap"] = base_relative_rotation
        cmd["rotation_deg"] = final_relative_rotation
        cmd["rotation_fine_offset_deg"] = piece_rot_fine
        cmd["rotation_snap_delta_deg"] = total_visual_delta_deg
        cmd["rotation_snap_applied"] = abs(total_visual_delta_deg) > 1e-9
        cmd["piece_abs_before_deg"] = piece_abs_before_deg
        cmd["piece_abs_target_deg"] = piece_abs_target_deg
        cmd["piece_abs_cardinal_delta_deg"] = piece_abs_cardinal_delta_deg

        if piece_abs_before_deg is not None:
            print(
                f"[A5 ABS SNAP] Piece {cmd['piece_id']}: "
                f"abs_before={piece_abs_before_deg:.2f}° -> abs_target={piece_abs_target_deg:.2f}° "
                f"(delta={piece_abs_cardinal_delta_deg:+.2f}°), "
                f"relative={base_relative_rotation:.2f}° -> final={final_relative_rotation:.2f}°"
            )
        elif cmd["rotation_snap_applied"]:
            print(
                f"[ROT FINE] Piece {cmd['piece_id']}: "
                f"relative={base_relative_rotation:.2f}° -> final={final_relative_rotation:.2f}° "
                f"(delta={total_visual_delta_deg:+.2f}°)"
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

    # Nach der Grundplatzierung einen kleinen Abstand zwischen den Teilen erzeugen.
    # Das verändert nur die Placepunkte, nicht die Pickpunkte.
    apply_inter_piece_gap(robot_commands, best)
    apply_grid_overlap_separation(robot_commands, min_gap_mm=3.0)

    # Danach die gesamte Lösung bei Bedarf zurück in den durch die Firmware
    # erlaubten Roboterbereich schieben. Kein Einzelpunkt-Clamping.
    safety_dx, safety_dy = keep_place_points_inside_robot_bounds(robot_commands)
    best["place_safety_shift_robot"] = (safety_dx, safety_dy)

    print(
        f"[A5 ALIGN] A5_size=({A5_WIDTH_MM:.1f}, {A5_HEIGHT_MM:.1f}) mm, "
        f"deskew={best['base_deskew_deg']:.2f}°, "
        f"extra={best['extra_rot']}°, "
        f"layout_rot={best['layout_rot']:.2f}°, "
        f"scale={best['scale']:.4f} mm/px, "
        f"puzzle_size=({best['final_w']:.1f}, {best['final_h']:.1f}) mm, "
        f"offset=({offset_x:.1f}, {offset_y:.1f}) mm, "
        f"gap={A5_INTER_PIECE_GAP_MM:.1f} mm, "
        f"snap_cardinal={A5_SNAP_ROTATIONS_TO_CARDINAL} force={A5_FORCE_CARDINAL_ROTATIONS} tol={A5_CARDINAL_SNAP_TOLERANCE_DEG:.1f}°, "
        f"place_robot_fine=({PLACE_ROBOT_FINE_OFFSET_X_MM:.1f}, {PLACE_ROBOT_FINE_OFFSET_Y_MM:.1f}) mm, "
        f"safety_shift_robot=({best['place_safety_shift_robot'][0]:+.1f}, {best['place_safety_shift_robot'][1]:+.1f}) mm"
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

    # Vor dem Solver sichern, weil solve_puzzle() die Piece-Pixel mutiert.
    # Diese Snapshots werden später verwendet, um die echte benötigte Rotation
    # per Form-Matching zu berechnen statt blind TRANSFORM_REPORT zu vertrauen.
    initial_piece_points = snapshot_piece_points_by_id(puzzle.pieces_)

    print("[RUN] Löse Puzzle...")
    puzzle.solve_puzzle()
    print("[RUN] Puzzle gelöst.")

    # Aktuelle Solver-Debugbilder speichern. Weil der Debug-Ordner am Start
    # geleert wurde, können hier keine alten 15_debug/18_debug/... Bilder liegen bleiben.
    for i, debug_img in enumerate(puzzle.get_debug_images(), start=10):
        save_debug_image(f"{i:02d}_debug.png", debug_img)

    solved_pick_centers = detect_pick_centers_on_solved_puzzle(puzzle.pieces_)
    solved_piece_points = snapshot_piece_points_by_id(puzzle.pieces_)
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

        # Rotation robuster bestimmen: nicht mehr nur TRANSFORM_REPORT verwenden,
        # sondern ursprüngliche und gelöste Teilform relativ zum Greifpunkt matchen.
        if pick_center is not None and solved_pick is not None:
            shape_rotation = estimate_piece_rotation_from_shape(
                initial_piece_points.get(int(report["piece_id"])),
                (float(pick_center["col"]), float(pick_center["row"])),
                solved_piece_points.get(int(report["piece_id"])),
                (float(solved_pick["col"]), float(solved_pick["row"])),
                piece_id=int(report["piece_id"]),
            )
            if shape_rotation is not None:
                cmd["rotation_from_transform_report_deg"] = cmd["rotation_deg"]
                cmd["rotation_deg"] = shape_rotation
                cmd["rotation_source"] = "shape_match"
            else:
                cmd["rotation_source"] = "transform_report"
        else:
            cmd["rotation_source"] = "transform_report"

        stats = solved_piece_stats.get(int(report["piece_id"]))
        if stats is not None:
            cmd["solved_bbox_center_px"] = stats["bbox_center_px"]

        robot_commands.append(cmd)

    robot_commands = deduplicate_robot_commands_by_piece(robot_commands)
    robot_commands = sorted(
        robot_commands,
        key=lambda x: x["piece_id"],
    )
    solution_points_px = get_solution_points_px(puzzle.pieces_)
    align_info = align_solution_to_a5(robot_commands, solution_points_px, solved_piece_points)
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
        # Vor dem ersten Roboter-MOVE alles prüfen, damit ein ungültiger Punkt
        # nicht erst mitten im Ablauf auffällt, wenn bereits ein Teil angesaugt ist.
        validate_robot_plan(robot_commands)
        send_to_robot(robot_commands)
    else:
        # Auch ohne Senden prüfen, damit Grenzfehler bereits im Debuglauf sichtbar sind.
        validate_robot_plan(robot_commands)
        print("\n[ROBOT] SEND_TO_ROBOT=False -> Es wurde nichts an den Roboter gesendet.")

    log_duration("total run", t_total)
    print("[RUN] Fertig.")


if __name__ == "__main__":
    freeze_support()
    main()


