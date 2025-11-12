import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageTk, ImageOps
import cv2

from .frames import TopFrame, NavigationFrame

class PuzzleSolverSimulator(tk.Canvas):
    """
    The main application class.
    It manages the application state, orchestrates the UI frames,
    and handles the core logic.
    """
    def __init__(self, root, extractor):
        self.root = root
        self.root.title("PuzzleSolver Simulator")

        self.dimensions = (1200, 700)
        self.image_dimensions = (self.dimensions[0] - 100, self.dimensions[1] - 120)

        self.root.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")

        # Row 0 (top_frame) - no vertical expansion
        self.root.grid_rowconfigure(0, weight=0)
        # Row 1 (log_view, image_label) - expands vertically
        self.root.grid_rowconfigure(1, weight=1)
        # Row 2 (nav_frame) - no vertical expansion
        self.root.grid_rowconfigure(2, weight=0)

        # Column 0 (log_view) - no horizontal expansion
        self.root.grid_columnconfigure(0, weight=0)
        # Column 1 (image_label) - expands horizontally
        self.root.grid_columnconfigure(1, weight=1)

        # Extractor
        self.extractor = extractor

        # State Variables
        self.debug_images_np = []   # Holds raw numpy images
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

        # UI setup
        self.top_frame = TopFrame(root)
        # Grid layout: Top row, spans 2 columns, sticks to East-West (horizontal)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=1)

        # Log view
        self.log_view = tk.Text(
            root,
            wrap=tk.WORD,
            width=50,
            font=("Helvetica", 14),
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0,
            padx=5,
            pady=5,
            spacing1=5
        )
        # Grid layout: Center row, first column, sticks to all sides
        self.log_view.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        # Image display area
        self.image_label = tk.Label(root)
        # Grid layout: Center row, second column, sticks to all sides
        self.image_label.grid(row=1, column=1, sticky="nsew", pady=10, padx=10)

        # Navigation frame
        self.nav_frame = NavigationFrame(
            root,
            load_callback=self.load_and_process_image,
            prev_callback=self.show_prev_step,
            next_callback=self.show_next_step
        )
        # Grid layout: Bottom row, spans 2 columns, sticks to East-West
        self.nav_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=1)

        self.set_title("PuzzleSolver Simulator | Gruppe 19")
        self.add_log_message("Simulator gestarted")
        self.add_log_message("Bitte ein Bild laden, um zu starten.")

    def load_and_process_image(self):
        """
        Loads an image, calls the extractor, and populates the simulator.
        """

        # Reset state
        self.clear_log()
        self.max_step = 0
        self.current_step = 0

        file_path = fd.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        # if not file_path:
        #     return

        # try:
        #     image = cv2.imread(file_path)
        # except Exception as e:
        #     self.add_log_message(f"Fehler beim Laden des Bildes: {e}")
        #     return

        try:
            # Call the extractor
            _pieces, _transforms, self.debug_images_np = self.extractor.extract_pieces_and_transformations(image=None, debug=True)
        except Exception as e:
            self.add_log_message(f"Fehler bei der Bildverarbeitung: {e}")
            return

        if not self.debug_images_np:
            return

        # Set initial state
        self.current_step = 0
        self.update_display()

    def update_display(self):
        """
        Updates the image and labels to reflect the current step.
        Manages the state of the navigation buttons.
        """
        if self.current_step < 0:
            return

        # Update Image
        img_pil = Image.fromarray(self.debug_images_np[self.current_step])

        # Calculate new dimensions while maintaining aspect ratio
        # We use the actual label size for containment
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # Use a default size if window not drawn yet
        if label_width <= 1 or label_height <= 1:
            label_width, label_height = self.image_dimensions

        img_pil = ImageOps.contain(img_pil, (label_width, label_height))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk # Keep a reference

        # Update Step Label text via the TopFrame's method
        step_desc = self.step_descriptions[self.current_step]
        total_steps = len(self.debug_images_np)
        self.set_title(f"Schritt {self.current_step + 1}/{total_steps}: {step_desc}")
        self.add_log_message(step_desc)

        if self.current_step == total_steps - 1:
            self.log_transformation_results()

        # Update Button States via the NavigationFrame's method
        can_go_prev = self.current_step > 0
        can_go_next = self.current_step < total_steps - 1
        self.nav_frame.set_button_states(can_go_prev, can_go_next)

    def show_next_step(self):
        """Moves to the next step if possible."""
        total_steps = len(self.debug_images_np)
        if self.current_step < total_steps - 1:
            next_step = self.current_step + 1
            self.max_step = max(self.max_step, next_step)
            self.current_step = next_step
            self.update_display()

    def show_prev_step(self):
        """Moves to the previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()

    def add_log_message(self, message):
        # If we are stepping back, do not add new log messages
        if self.max_step > self.current_step:
            return

        self.log_view.configure(state=tk.NORMAL)
        self.log_view.insert(tk.INSERT, f"[LOG]: {message}\n")
        self.log_view.configure(state=tk.DISABLED)

    def clear_log(self):
        self.log_view.configure(state=tk.NORMAL)
        self.log_view.delete(1.0, tk.END)
        self.log_view.configure(state=tk.DISABLED)

    def log_transformation_results(self):
        self.add_log_message("Puzzleteil 1:")
        self.add_log_message(" - Bewegung: (1222, 1222) -> (12445, 33333)")
        self.add_log_message(" - Rotation: 30°")
        self.add_log_message("Puzzleteil 2:")
        self.add_log_message(" - Bewegung: (222, 222) -> (4445, 5555)")
        self.add_log_message(" - Rotation: 15°")
        self.add_log_message("Puzzleteil 3:")
        self.add_log_message(" - Bewegung: (555, 666) -> (7777, 8888)")
        self.add_log_message(" - Rotation: 45°")
        self.add_log_message("Puzzleteil 4:")
        self.add_log_message(" - Bewegung: (999, 111) -> (1212, 1313)")
        self.add_log_message(" - Rotation: 60°")

    def set_title(self, title):
        self.top_frame.set_text(title)
