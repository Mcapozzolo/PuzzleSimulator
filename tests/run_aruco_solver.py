import os
import sys
import cv2
from multiprocessing import freeze_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.Vision import VisionPipeline
from solver.Puzzle.Puzzle import Puzzle


DEBUG_DIR = os.path.join(PROJECT_ROOT, "assets", "DEBUG_ARUCO")
TEMP_DIR = os.path.join(PROJECT_ROOT, "assets", "TEST")
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def main():
    image_path = r"assets\Bilder aruco marker\test\Image (1).png"

    pipeline = VisionPipeline(
        marker_length_mm=20.0,
        workspace_output_size_px=(1200, 800),
        workspace_mm_size=(400.0, 300.0),
        aruco_ids=(0, 1, 2, 3),
    )

    result = pipeline.process_image_from_path(image_path)

    warped = result["warped_workspace"]
    if warped is None:
        raise ValueError("Warped workspace ist None -> ArUco Fehler")

    h, w = warped.shape[:2]
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.12)
    warped_inner = warped[margin_y:h - margin_y, margin_x:w - margin_x].copy()

    cv2.imwrite(os.path.join(DEBUG_DIR, "01_aruco_debug.png"), result["aruco_debug"])
    cv2.imwrite(os.path.join(DEBUG_DIR, "02_warped_workspace.png"), warped)
    cv2.imwrite(os.path.join(DEBUG_DIR, "03_warped_inner.png"), warped_inner)

    warped_path = os.path.join(TEMP_DIR, "warped_workspace_temp.png")
    cv2.imwrite(warped_path, warped_inner)

    print(f"[ARUCO] Solver input: {warped_path}")

    puzzle = Puzzle(warped_path, log_fn=print)

    puzzle.extract_pieces()
    print(f"[ARUCO] Extrahierte Teile: {len(puzzle.pieces_)}")

    if len(puzzle.pieces_) == 0:
        raise ValueError("Keine Puzzleteile erkannt")

    puzzle.solve_puzzle()
    print("[ARUCO] Puzzle erfolgreich gelöst")

    for i, img in enumerate(puzzle.get_debug_images()):
        out = os.path.join(DEBUG_DIR, f"{i + 10:02d}_puzzle_debug.png")
        cv2.imwrite(out, img)
        print(f"[DEBUG] Saved {out}")


if __name__ == "__main__":
    freeze_support()
    main()