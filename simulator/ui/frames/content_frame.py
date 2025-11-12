import tkinter as tk


class ContentFrame(tk.Frame):
    """
    A frame to contain the log view and the image display area.
    It manages its own internal grid layout.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Column 0 (log_view) - fixed width
        self.grid_columnconfigure(0, weight=0)
        # Column 1 (image_label) - expands horizontally
        self.grid_columnconfigure(1, weight=1)
        # Row 0 (main content) - expands vertically
        self.grid_rowconfigure(0, weight=1)

        # Log view
        self.log_view = tk.Text(
            self,  # Parent is this frame
            wrap=tk.WORD,
            width=50,
            font=("Helvetica", 14),
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0,
            padx=5,
            pady=5,
            spacing1=5,
        )
        # Grid layout: Center row, first column
        self.log_view.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=10)

        # Image display area
        self.image_label = tk.Label(self)  # Parent is this frame
        # Grid layout: Center row, second column
        self.image_label.grid(row=0, column=1, sticky="nsew", pady=10, padx=(0, 10))

    def append_log(self, message):
        """Append a message to the log view."""
        self.log_view.config(state=tk.NORMAL)
        self.log_view.insert(tk.END, f"[LOG]: {message}\n")
        self.log_view.see(tk.END)
        self.log_view.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log view."""
        self.log_view.config(state=tk.NORMAL)
        self.log_view.delete(1.0, tk.END)
        self.log_view.config(state=tk.DISABLED)

    def update_image(self, img_tk):
        """Update the displayed image."""
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def get_image_dimensions(self):
        """Return the current dimensions of the image display area."""
        self.image_label.update_idletasks()
        return (self.image_label.winfo_width(), self.image_label.winfo_height())
