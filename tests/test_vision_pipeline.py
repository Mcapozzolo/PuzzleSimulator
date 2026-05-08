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
            "type": "TODO",
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
            "type": "TODO",
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