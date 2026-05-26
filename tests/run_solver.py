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