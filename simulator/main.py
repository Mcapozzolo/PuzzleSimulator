import tkinter as tk

from extractor import MockExtractor
from .ui.simulator import PuzzleSolverSimulator


if __name__ == "__main__":
    """
    Main entry point for the application.
    """
    root = tk.Tk()

    extractor = MockExtractor()
    app = PuzzleSolverSimulator(root, extractor)

    root.mainloop()
