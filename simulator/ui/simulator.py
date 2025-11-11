import tkinter as tk
from PIL import Image, ImageTk, ImageOps

from .frames import TopFrame, NavigationFrame

class PuzzleSolverSimulator:
    """
    The main application class.
    It manages the application state, orchestrates the UI frames,
    and handles the core logic.
    """
    def __init__(self, root, extractor):
        self.root = root
        self.root.title("PuzzleSolver Simulator")

        self.DIMENSIONS = (1000, 600)

        self.root.geometry(f"{self.DIMENSIONS[0]}x{self.DIMENSIONS[1]}")

        self.IMAGE_DIMENSIONS = (self.DIMENSIONS[0] - 100, self.DIMENSIONS[1] - 120)

        # Extractor
        self.extractor = extractor

        # State Variables
        self.debug_images_np = []   # Holds raw numpy images
        self.debug_images_tk = []   # Holds PhotoImage objects
        self.step_descriptions = [] # Holds the description for each step
        self.current_step = -1

        # UI setup
        self.top_frame = TopFrame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=1)
        self.top_frame.set_text("Bitte ein Bild laden, um zu starten.")

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
            # Get descriptions from the extractor
            self.step_descriptions = self.extractor.step_descriptions_
        except Exception as e:
            # Update UI on error
            self.top_frame.set_text(f"Fehler bei der Bildverarbeitung: {e}")
            return

        # Clear previous run
        self.debug_images_tk.clear()

        if not self.debug_images_np:
            return

        # Process and store PhotoImage objects
        for img in self.debug_images_np:
            # Resize image to fit the label
            img_pil = Image.fromarray(img)
            img_pil = ImageOps.contain(img_pil, self.IMAGE_DIMENSIONS)
            self.debug_images_tk.append(ImageTk.PhotoImage(img_pil))

        # Set initial state
        self.current_step = 0
        self.update_display()

    def update_display(self):
        """
        Updates the image and labels to reflect the current step.
        Manages the state of the navigation buttons.
        """
        if self.current_step < 0 or not self.debug_images_tk:
            return

        # Update Image
        self.image_label.config(image=self.debug_images_tk[self.current_step])

        # Update Step Label text via the TopFrame's method
        step_desc = self.step_descriptions[self.current_step]
        total_steps = len(self.debug_images_tk)
        self.top_frame.set_text(f"Schritt {self.current_step + 1}/{total_steps}: {step_desc}")

        # Update Button States via the NavigationFrame's method
        can_go_prev = self.current_step > 0
        can_go_next = self.current_step < total_steps - 1
        self.nav_frame.set_button_states(can_go_prev, can_go_next)

    def show_next_step(self):
        """Moves to the next step if possible."""
        total_steps = len(self.debug_images_tk)
        if self.current_step < total_steps - 1:
            self.current_step += 1
            self.update_display()

    def show_prev_step(self):
        """Moves to the previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()