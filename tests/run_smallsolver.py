import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solver.Puzzle.Puzzle import Puzzle
def main():
    parser = argparse.ArgumentParser(description="Run the agglomerative SmallSolver on one image.")
    parser.add_argument("image")
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    p = Puzzle(args.image, log_fn=print)
    p.extract_pieces()
    ok = p.solve_puzzle_small(fallback=not args.no_fallback)
    print("small solver ok:", ok)


if __name__ == "__main__":
    main()
