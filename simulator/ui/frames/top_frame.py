import tkinter as tk


class TopFrame(tk.Frame):
    """
    The top frame of the simulator, which displays the
    current step description.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create the label for step descriptions
        self.step_label = tk.Label(self, font=("Helvetica", 18))
        self.step_label.pack(side=tk.LEFT, pady=10)

    def set_text(self, text):
        """Update the label's text."""
        self.step_label.config(text=text)
