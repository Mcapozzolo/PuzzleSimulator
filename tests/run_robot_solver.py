import os
import sys
import cv2
from multiprocessing import freeze_support

print("[BOOT] run_robot_solver.py wurde gestartet", flush=True)

# =========================
# CONFIG
# =========================

USE_CAMERA = False
CAMERA_INDEX = 0

IMAGE_PATH = r"assets\Bilder aruco marker\test\Image (2).jpg"

WORKSPACE_SIZE_PX = (1200, 800)

# WICHTIG:
# Das ist die echte physische Grösse der Fläche zwischen den 4 ArUco-Workspace-Ecken.
# Also nicht nur A4, sondern A4 plus Marker-/Randbereich, falls die Marker ausserhalb A4 liegen.
WORKSPACE_SIZE_MM = (400.0, 300.0)

CROP_MARGIN_RATIO_X = 0.02
CROP_MARGIN_RATIO_Y = 0.02

SAFE_Z_MM = 50.0
PICK_Z_MM = 5.0

# =========================
# ROBOT KOORDINATEN
# =========================

# Roboterkoordinate der ArUco-Workspace-Ecke A0.
# A0 = obere linke Ecke des gewarpten ArUco-Workspace, NICHT Marker-Mitte.
# Diese Werte einmal manuell messen:
# Roboter auf A0 fahren -> X/Y ablesen -> hier eintragen.
PICK_OFFSET_X_MM = 0.0
PICK_OFFSET_Y_MM = 0.0

# Roboterkoordinate der A5-Zielfläche.
# Da euer Arduino-Nullpunkt bei der A5-Fläche liegt, hier entsprechend eintragen.
A5_ORIGIN_X_MM = 0.0
A5_ORIGIN_Y_MM = 0.0

# Damit das gelöste Puzzle nicht exakt auf der A5-Ecke beginnt,
# sondern etwas nach innen verschoben liegt.
A5_PUZZLE_OFFSET_X_MM = 20.0
A5_PUZZLE_OFFSET_Y_MM = 20.0

SEND_TO_ROBOT = False
ROBOT_PORT = "COM3"

DEBUG_SAVE = True

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
    """
    Pick-Koordinate:
    ArUco-/A4-Workspace -> Roboterkoordinatensystem.
    """
    return (
        PICK_OFFSET_X_MM + x_mm,
        PICK_OFFSET_Y_MM + y_mm,
    )


def place_to_robot(x_mm, y_mm):
    """
    Place-Koordinate:
    Solver-Zielposition -> A5-Fläche im Roboterkoordinatensystem.
    """
    return (
        A5_ORIGIN_X_MM + A5_PUZZLE_OFFSET_X_MM + x_mm,
        A5_ORIGIN_Y_MM + A5_PUZZLE_OFFSET_Y_MM + y_mm,
    )


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

    out = os.path.join(DEBUG_DIR, "robot_pick_coordinates.png")
    cv2.imwrite(out, debug)
    print(f"[DEBUG] Saved robot overlay: {out}")

def normalize_rotation_deg(angle_deg):
    return round(((angle_deg + 180) % 360) - 180, 2)


def build_pick_place_sequence(cmd):
    pick_x, pick_y = pick_to_robot(
        cmd["pick_x_mm"],
        cmd["pick_y_mm"],
    )

    place_x, place_y = place_to_robot(
        cmd["place_x_mm"],
        cmd["place_y_mm"],
    )

    rotation_deg = normalize_rotation_deg(cmd["rotation_deg"])

    return [
        {
            "description": "above_pick",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": SAFE_Z_MM,
        },
        {
            "description": "down_to_pick",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": PICK_Z_MM,
        },
        {
            "description": "suction_on",
            "type": "suction",
        },
        {
            "description": "lift_piece",
            "x_mm": pick_x,
            "y_mm": pick_y,
            "z_mm": SAFE_Z_MM,
        },
        {
            "description": "above_place",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": SAFE_Z_MM,
            "rotation_deg": rotation_deg,
        },
        {
            "description": "down_to_place",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": PICK_Z_MM,
            "rotation_deg": rotation_deg,
        },
        {
            "description": "suction_off",
            "type": "suction",
        },
        {
            "description": "lift_after_place",
            "x_mm": place_x,
            "y_mm": place_y,
            "z_mm": SAFE_Z_MM,
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

            if step.get("type") == "suction":
                print(f"  {step['description']}")
                continue

            print(
                f"  {step['description']}: "
                f"X={step['x_mm']:.2f} mm, "
                f"Y={step['y_mm']:.2f} mm, "
                f"Z={step['z_mm']:.2f} mm, "
                f"C={step.get('rotation_deg', 0.0):.2f}°"
            )


def send_to_robot(robot_commands):
    robot = RobotInterface(port=ROBOT_PORT, send_units="cm")

    try:
        print("[ROBOT] READY")
        robot.ready()

        for cmd in robot_commands:
            print(f"[ROBOT] Piece {cmd['piece_id']}")

            for step in build_pick_place_sequence(cmd):
                print(
                    f"[ROBOT MOVE] {step['description']} "
                    f"X={step['x_mm']:.2f}, "
                    f"Y={step['y_mm']:.2f}, "
                    f"Z={step['z_mm']:.2f}"
                )

                if step["description"] == "suction_on":
                    robot.suction_on()
                    robot.wait_until_idle()
                    continue

                if step["description"] == "suction_off":
                    robot.suction_off()
                    robot.wait_until_idle()
                    continue

                robot.move_xyzc_mm_and_wait(
                    step["x_mm"],
                    step["y_mm"],
                    step["z_mm"],
                    step.get("rotation_deg", 0.0),
                )

        print("[ROBOT] FINISH")
        robot.finish()

    finally:
        robot.close()


def main():
    print("[RUN] Starte Vision Pipeline...")

    pipeline = VisionPipeline(
        marker_length_mm=20.0,
        workspace_output_size_px=WORKSPACE_SIZE_PX,
        workspace_mm_size=WORKSPACE_SIZE_MM,
        aruco_ids=(0, 1, 2, 3),
    )

    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden.")

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError("Kamera konnte kein Bild aufnehmen.")

        cv2.imwrite(os.path.join(DEBUG_DIR, "00_camera_input.png"), frame)
        result = pipeline.process_image(frame)
    else:
        result = pipeline.process_image_from_path(IMAGE_PATH)

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

    if DEBUG_SAVE:
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_aruco_debug.png"), result["aruco_debug"])
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_warped_workspace.png"), warped)
        cv2.imwrite(os.path.join(DEBUG_DIR, "03_warped_inner.png"), warped_inner)

    transformation_logs = []

    def record_log(msg):
        print(msg)

        if msg.startswith("TRANSFORM_REPORT"):
            transformation_logs.append(msg)

    puzzle = Puzzle(warped_path, log_fn=record_log)

    print("[RUN] Extrahiere Puzzleteile...")
    puzzle.extract_pieces()

    print(f"[RUN] Extrahierte Teile: {len(puzzle.pieces_)}")

    initial_pick_centers = {}

    for piece in puzzle.pieces_:
        minX, minY, maxX, maxY = piece.get_bbox()

        cx = (minX + maxX) / 2
        cy = (minY + maxY) / 2

        # get_bbox liefert Array-Koordinaten: row/col
        initial_pick_centers[piece.id] = {
            "row": cx,
            "col": cy,
        }

    if len(puzzle.pieces_) == 0:
        raise ValueError("Keine Puzzleteile erkannt!")

    print("[RUN] Löse Puzzle...")
    puzzle.solve_puzzle()
    print("[RUN] Puzzle gelöst.")

    if DEBUG_SAVE:
        for i, img in enumerate(puzzle.get_debug_images()):
            out = os.path.join(DEBUG_DIR, f"{10 + i:02d}_debug.png")
            cv2.imwrite(out, img)
            print(f"[DEBUG] Saved puzzle debug: {out}")

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

        robot_commands.append(cmd)

    robot_commands = sorted(
        robot_commands,
        key=lambda x: x["piece_id"],
    )

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

    print("[RUN] Fertig.")


if __name__ == "__main__":
    freeze_support()
    main()