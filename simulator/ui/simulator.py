import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageTk, ImageOps
import threading
from solver.Puzzle.Puzzle import Puzzle

from .frames.top_frame import TopFrame
from .frames.content_frame import ContentFrame
from .frames.navigation_frame import NavigationFrame


class PuzzleSolverSimulator(tk.Canvas):
    def __init__(self, root, extractor):
        self.root = root
        self.extractor = extractor
        self.dimensions = (1200, 700)
        self.transformation_logs = []

        def _record_transformation(msg: str):
            if msg.startswith("TRANSFORM_REPORT"):
                self.transformation_logs.append(msg)
            else:
                self.add_log_message(msg)

        self.record_transformation = _record_transformation

        self.root.title("PuzzleSolver Simulator")
        self.root.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")

        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

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

        self.debug_images_np = []
        self.step_descriptions = [
            "Eingabebild geladen",
            "Threshholding angewendet",
            "Konturen erkannt",
            "Puzzle Kanten erkannt",
            "Puzzle zusammensetzen",
        ]
        self.current_step = 0
        self.max_step = 0

        self.set_title("PuzzleSolver Simulator | Gruppe 19")
        self.add_log_message("Simulator gestarted")
        self.add_log_message("Bitte ein Bild laden, um zu starten.")

    def update_display(self):
        width, height = self.content_frame.get_image_dimensions()
        img = Image.fromarray(self.debug_images_np[self.current_step])
        img = ImageOps.contain(img, (width, height))
        img = ImageTk.PhotoImage(img)
        self.content_frame.update_image(img)

        total_steps = len(self.debug_images_np)
        if self.current_step > 4:
            self.set_title(f"Schritt {self.current_step + 1}/{total_steps}: {self.step_descriptions[4]}")
        else:
            self.set_title(f"Schritt {self.current_step + 1}/{total_steps}: {self.step_descriptions[self.current_step]}")

        can_go_prev = self.current_step > 0
        can_go_next = self.current_step < total_steps - 1
        self.nav_frame.set_button_states(can_go_prev, can_go_next)

    def _process_image_task(self, file_path):
        try:
            puzzle = Puzzle(file_path, log_fn=self.record_transformation)

            self.add_log_message(f"Bild geladen: {file_path}")
            self.add_log_message("Bild wird verarbeitet.")

            puzzle.extract_pieces()

            self.add_log_message("Puzzleteile wurden extrahiert.")
            self.add_log_message("Puzzle wird gelöst...")

            puzzle.solve_puzzle()

            self.add_log_message("Puzzle wurde erfolgreich gelöst.")

            self.debug_images_np = puzzle.get_debug_images()

            self.log_transformation_results()

        except Exception as e:
            self.add_log_message(f"Fehler bei der Bildverarbeitung: {e}")

        finally:
            self.nav_frame.load_button.config(state=tk.NORMAL)

            if self.debug_images_np:
                self.root.after(0, self.update_display)

    def on_load_and_process_image(self):
        if len(self.debug_images_np) > 0:
            self.content_frame.clear_log()

        self.max_step = 0
        self.current_step = 0
        self.transformation_logs.clear()

        file_path = fd.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )

        if not file_path:
            self.add_log_message("Kein Bild ausgewählt.")
            return

        self.nav_frame.load_button.config(state=tk.DISABLED)

        self.add_log_message("Bild wird geladen und verarbeitet...")

        process_thread = threading.Thread(
            target=self._process_image_task,
            args=(file_path,),
            daemon=True
        )

        process_thread.start()

    def on_show_next_step(self):
        total_steps = len(self.debug_images_np)
        if self.current_step < total_steps - 1:
            next_step = self.current_step + 1
            self.max_step = max(self.max_step, next_step)
            self.current_step = next_step
            self.update_display()

    def on_show_prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()

    def add_log_message(self, message):
        if not message.strip():
            return
        if self.max_step > self.current_step:
            return

        self.content_frame.append_log(message)

    def set_title(self, title):
        self.top_frame.set_text(title)

    def log_transformation_results(self):
        reports = {}

        for msg in self.transformation_logs:
            if msg.startswith("TRANSFORM_REPORT"):
                parts = msg.split()
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
            self.add_log_message("")