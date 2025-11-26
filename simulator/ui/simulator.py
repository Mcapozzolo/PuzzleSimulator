import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageTk, ImageOps
import cv2
from solver.Puzzle.Puzzle import Puzzle

from .frames.top_frame import TopFrame
from .frames.content_frame import ContentFrame
from .frames.navigation_frame import NavigationFrame


class PuzzleSolverSimulator(tk.Canvas):
    """
    The main application class.
    It manages the application state, orchestrates the UI frames,
    and handles the core logic.
    """

    def __init__(self, root, extractor):
        self.root = root
        self.extractor = extractor
        self.dimensions = (1200, 700)
        # collected transformation/log messages coming from solver/mover
        self.transformation_logs = []
        # callback to hand to Puzzle/solver/mover
        def _record_transformation(msg: str):
            # Speziell: TRANSFORM_REPORT sammeln wir nur
            if msg.startswith("TRANSFORM_REPORT"):
                self.transformation_logs.append(msg)
            else:
                # normale Logs sofort anzeigen
                self.add_log_message(msg)
  

        self.record_transformation = _record_transformation

        self.root.title("PuzzleSolver Simulator")
        self.root.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")

        # Row 0 (top_frame) - no vertical expansion
        self.root.grid_rowconfigure(0, weight=0)
        # Row 1 (content_frame) - expands vertically
        self.root.grid_rowconfigure(1, weight=1)
        # Row 2 (nav_frame) - no vertical expansion
        self.root.grid_rowconfigure(2, weight=0)

        # Column 0 (content_frame - log_view) - no horizontal expansion
        self.root.grid_columnconfigure(0, weight=0)
        # Column 1 (content_frame - image_label) - expands horizontally
        self.root.grid_columnconfigure(1, weight=1)

        # UI setup
        self.top_frame = TopFrame(root)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=1)

        self.content_frame = ContentFrame(root)
        self.content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.nav_frame = NavigationFrame(
            root,
            load_callback=self.on_load_and_process_image,
            prev_callback=self.on_show_prev_step,
            next_callback=self.on_show_next_step,
        )
        self.nav_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=1)

        # State variables
        self.debug_images_np = []  # Holds raw numpy images
        self.step_descriptions = [
            "Eingabebild geladen",
            "Graustufen-Konvertierung angewendet",
            "Konturen erkannt",
            "Puzzleteile erkannt",
            "Puzzleteile extrahiert",
            "Puzzleteile korrigiert",
            "Ecken erkannt",
            "Puzzle zusammengesetzt",
        ]
        self.current_step = 0
        self.max_step = 0

        # Initial UI state
        self.set_title("PuzzleSolver Simulator | Gruppe 19")
        self.add_log_message("Simulator gestarted")
        self.add_log_message("Bitte ein Bild laden, um zu starten.")

    def update_display(self):
        """
        Updates the image and labels to reflect the current step.
        Manages the state of the navigation buttons.
        """

        # Update image
        width, height = self.content_frame.get_image_dimensions()
        img = Image.fromarray(self.debug_images_np[self.current_step])
        img = ImageOps.contain(img, (width, height))
        img = ImageTk.PhotoImage(img)
        self.content_frame.update_image(img)

        # Update title and log
        # step_desc = self.step_descriptions[self.current_step]
        total_steps = len(self.debug_images_np)
        self.set_title(f"Schritt {self.current_step + 1}/{total_steps}")
        # self.set_title(f"Schritt {self.current_step + 1}/{total_steps}: {step_desc}")
        # self.add_log_message(step_desc)

        # Update button states
        can_go_prev = self.current_step > 0
        can_go_next = self.current_step < total_steps - 1
        self.nav_frame.set_button_states(can_go_prev, can_go_next)

    def on_load_and_process_image(self):
        """
        Loads an image, calls the extractor, and populates the simulator.
        """

        # Reset state
        self.content_frame.clear_log()
        self.max_step = 0
        self.current_step = 0
        # clear previous transformation logs
        self.transformation_logs.clear()

        file_path = fd.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )

        if not file_path:
            self.add_log_message("Kein Bild ausgewählt.")
            return

        try:
            # Call the extractor and try to provide a logging callback
            try:
                puzzle = Puzzle(file_path, log_fn=self.record_transformation)
            except TypeError:
                puzzle = Puzzle(file_path)
                # try attaching common attribute names if available
                if hasattr(puzzle, "set_log_fn"):
                    try:
                        puzzle.set_log_fn(self.record_transformation)
                    except Exception:
                        pass
                elif hasattr(puzzle, "log_fn"):
                    try:
                        puzzle.log_fn = self.record_transformation
                    except Exception:
                        pass
            self.add_log_message(f"Bild geladen: {file_path}")
            self.add_log_message("Bild wird verarbeitet.")

            puzzle.extract_pieces()
            self.add_log_message("Puzzleteile wurden extrahiert")
            self.add_log_message("Puzzle wird gelöst.")

            puzzle.solve_puzzle()

            self.debug_images_np = puzzle.get_debug_images()
            # self.debug_images_np = puzzle.get_debug_images()
            # _pieces, _transforms, self.debug_images_np = (
            #     self.extractor.extract_pieces_and_transformations(
            #         image=None, debug=True
            #     )
            # )
            self.log_transformation_results()

        except Exception as e:
            self.add_log_message(f"Fehler bei der Bildverarbeitung: {e}")

        if not self.debug_images_np:
            return

        # Set initial state
        self.update_display()

    def on_show_next_step(self):
        """Moves to the next step if possible."""
        total_steps = len(self.debug_images_np)
        if self.current_step < total_steps - 1:
            next_step = self.current_step + 1
            self.max_step = max(self.max_step, next_step)
            self.current_step = next_step
            self.update_display()

    def on_show_prev_step(self):
        """Moves to the previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()

    def add_log_message(self, message):
        if not message.strip():
            return
        # If we are stepping back, do not add new log messages
        if self.max_step > self.current_step:
            return

        self.content_frame.append_log(message)

    def set_title(self, title):
        self.top_frame.set_text(title)

    def log_transformation_results(self):
        # Alle TRANSFORM_REPORT-Zeilen einsammeln
        reports = {}

        for msg in self.transformation_logs:
            if msg.startswith("TRANSFORM_REPORT"):
                parts = msg.split()
                # Format:
                # TRANSFORM_REPORT <id> <x_old> <y_old> <x_new> <y_new> <dx> <dy> <rot>
                _, piece_id, x0, y0, x1, y1, dx, dy, rot = parts
                reports[int(piece_id)] = {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "dx": int(dx),
                    "dy": int(dy),
                    "rot": float(rot),
                }

        # Sortiert nach Puzzleteil-ID ausgeben
        for pid in sorted(reports.keys()):
            tf = reports[pid]
            self.add_log_message(f"Puzzleteil {pid}:")
            self.add_log_message(
                f" - Bewegung: ({tf['x0']}, {tf['y0']}) -> ({tf['x1']}, {tf['y1']})"
            )
            self.add_log_message(
                f" - Translation: dx={tf['dx']}, dy={tf['dy']}"
            )
            self.add_log_message(
                f" - Rotation: {tf['rot']:.1f}°"
            )
            self.add_log_message("")  # Leerzeile für Übersicht
