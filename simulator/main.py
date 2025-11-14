import os
import tkinter as tk

from extractor import MockExtractor
from .ui.simulator import PuzzleSolverSimulator


if __name__ == "__main__":
    """
    Main entry point for the application.
    """
    root = tk.Tk()

    os.environ["ZOLVER_TEMP_DIR"] = "resources/debug"
    extractor = MockExtractor()
    app = PuzzleSolverSimulator(root, extractor)

    root.mainloop()
