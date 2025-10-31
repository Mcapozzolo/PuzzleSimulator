import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageTk, ImageOps
import random

class PuzzleSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle Solver Proof of Concept")

        self.canvas_width = 1000
        self.canvas_height = 800
        self.piece_size = 100
        self.SCALE = (500, 500)

        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.solve_button = Button(btn_frame, text="Solve", command=self.solve_puzzle)
        self.solve_button.pack(side="left", padx=5)

        self.reset_button = Button(btn_frame, text="Reset", command=self.reset_puzzle)
        self.reset_button.pack(side="left", padx=5)

        # Store pieces
        self.pieces = []
        MARGIN = 50
        IMG_CENTER = self.SCALE[0] // 2
        self.original_positions = [
            (MARGIN + IMG_CENTER, MARGIN + IMG_CENTER),
            (MARGIN + IMG_CENTER + 220, MARGIN + 270),
            (MARGIN + 220, MARGIN + IMG_CENTER + 150),
            (MARGIN + IMG_CENTER + 190, MARGIN + IMG_CENTER + 170),
        ]
        self.original_angles = [0, 0, 0, 0]

        # Load or create mock pieces
        self.load_pieces()

        # Scatter them randomly at the start
        self.scatter_pieces()

    def load_pieces(self):
        """Make mock puzzle pieces as colored squares (replace with real images)."""
        image_paths = [
            "assets/pieces_with_bg_transparent/piece_4.png",
            "assets/pieces_with_bg_transparent/piece_3.png",
            "assets/pieces_with_bg_transparent/piece_2.png",
            "assets/pieces_with_bg_transparent/piece_1.png",
        ]
        for path in image_paths:
            img = Image.open(path)
            img = ImageOps.contain(img, self.SCALE)
            self.pieces.append({
                "image": img,
                "tk_img": None,
                "canvas_id": None,
                "angle": 0,
            })

    def scatter_pieces(self):
        """Place pieces randomly on canvas without overlap."""
        # Divide canvas into regions to avoid overlap
        cols = len(self.pieces)
        region_width = self.canvas_width // cols

        for i, piece in enumerate(self.pieces):
            # Random position within assigned region
            x = random.randint(region_width * i + self.piece_size//2,
                               region_width * (i+1) - self.piece_size//2)
            y = random.randint(self.piece_size//2, self.canvas_height - self.piece_size//2)

            # Random rotation
            angle = random.randint(0, 360)
            piece["angle"] = angle
            rotated = piece["image"].rotate(angle, expand=True)
            piece["tk_img"] = ImageTk.PhotoImage(rotated)

            if piece["canvas_id"] is None:
                piece["canvas_id"] = self.canvas.create_image(x, y, image=piece["tk_img"])
            else:
                self.canvas.itemconfig(piece["canvas_id"], image=piece["tk_img"])
                self.canvas.coords(piece["canvas_id"], x, y)

    def solve_puzzle(self):
        """Rotate and place pieces in correct order."""
        for i, piece in enumerate(self.pieces):
            # Target position
            x, y = self.original_positions[i]
            angle = self.original_angles[i]

            # Rotate back to correct orientation
            rotated = piece["image"].rotate(angle, expand=True)
            piece["tk_img"] = ImageTk.PhotoImage(rotated)

            # Update canvas image
            self.canvas.itemconfig(piece["canvas_id"], image=piece["tk_img"])
            self.canvas.coords(piece["canvas_id"], x, y)

    def reset_puzzle(self):
        """Scatter pieces again (like restart)."""
        self.scatter_pieces()


if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleSimulator(root)
    root.mainloop()
