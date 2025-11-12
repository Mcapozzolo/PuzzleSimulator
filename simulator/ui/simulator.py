import tkinter as tk
import tkinter.scrolledtext as st
from PIL import Image, ImageTk, ImageOps

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

        self.dimensions = (1000, 600)
        self.image_dimensions = (self.dimensions[0] - 100, self.dimensions[1] - 120)

        self.root.geometry(f"{self.dimensions[0]}x{self.dimensions[1]}")

        # Extractor
        self.extractor = extractor

        # State Variables
        self.debug_images_np = []   # Holds raw numpy images
        self.step_descriptions = [  # Holds the description for each step
            "Input image loaded",
            "Grayscale conversion applied",
            "Contours detected",
            "Pieces detected",
            "Pieces extracted",
            "Pieces corrected",
            "Corners detected",
            "Puzzle assembled",
        ]
        self.current_step = -1

        # UI setup
        self.top_frame = TopFrame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=1)
        self.top_frame.set_text("Bitte ein Bild laden, um zu starten.")

        # Log views
        self.log_view = st.ScrolledText(
            root,
            wrap=tk.WORD,
            width=40,
            height=int((self.image_dimensions[1] - 20) / 14),
            font=("Helvetica", 14),
            state=tk.DISABLED,
            borderwidth=0
        )
        self.log_view.pack(side=tk.LEFT, padx=10, pady=10)

        # Image display area
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Navigation frame
        self.nav_frame = NavigationFrame(
            root,
            load_callback=self.load_and_process_image,
            prev_callback=self.show_prev_step,
            next_callback=self.show_next_step
        )
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=1)

        self.add_log_message("LOG: Simulator started")
        self.add_log_message("LOG: Simulator started 2")

    def add_log_message(self, message):
        self.log_view.configure(state=tk.NORMAL)
        self.log_view.insert(tk.INSERT, message + '\n')
        self.log_view.configure(state=tk.DISABLED)

    def load_and_process_image(self):
        """
        Loads an image, calls the extractor, and populates the simulator.
        """

        # To enable file dialog, uncomment the section below

        # file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        # if not file_path:
        #     return
        # try:
        #     pil_image = Image.open(file_path)
        # except Exception as e:
        #     self.top_frame.set_text(f"Fehler beim Laden des Bildes: {e}")
        #     return

        try:
            # Call the extractor
            _pieces, _transforms, self.debug_images_np = self.extractor.extract_pieces_and_transformations(image=None, debug=True)
        except Exception as e:
            # Update UI on error
            self.top_frame.set_text(f"Fehler bei der Bildverarbeitung: {e}")
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
        img_pil = ImageOps.contain(img_pil, self.image_dimensions)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)

        # Update Step Label text via the TopFrame's method
        step_desc = self.step_descriptions[self.current_step]
        total_steps = len(self.debug_images_np)
        self.top_frame.set_text(f"Schritt {self.current_step + 1}/{total_steps}: {step_desc}")

        # Update Button States via the NavigationFrame's method
        can_go_prev = self.current_step > 0
        can_go_next = self.current_step < total_steps - 1
        self.nav_frame.set_button_states(can_go_prev, can_go_next)

    def show_next_step(self):
        """Moves to the next step if possible."""
        total_steps = len(self.debug_images_np)
        if self.current_step < total_steps - 1:
            self.current_step += 1
            self.update_display()

    def show_prev_step(self):
        """Moves to the previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()