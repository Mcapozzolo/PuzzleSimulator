import tkinter as tk

from .ui.simulator import PuzzleSolverSimulator


if __name__ == "__main__":
    """
    Main entry point for the application.
    """
    root = tk.Tk()

    app = PuzzleSolverSimulator(root)

    root.mainloop()
