import tkinter as tk

class TopFrame(tk.Frame):
    """
    The top frame of the simulator, which displays the
    current step description.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create the label for step descriptions
        self.step_label = tk.Label(self, font=("Helvetica", 16))
        self.step_label.pack(side=tk.LEFT, pady=10)

    def set_text(self, text):
        """Update the label's text."""
        self.step_label.config(text=text)

class NavigationFrame(tk.Frame):
    """
    The bottom navigation frame with buttons to load an image
    and navigate between processing steps.
    """
    def __init__(self, parent, load_callback, prev_callback, next_callback, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create buttons and link them to callback functions
        self.load_button = tk.Button(self, text="Bild laden", command=load_callback)
        self.load_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = tk.Button(self, text="Nächster Schritt", command=next_callback, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=10)

        self.prev_button = tk.Button(self, text="Vorheriger Schritt", command=prev_callback, state=tk.DISABLED)
        self.prev_button.pack(side=tk.RIGHT, padx=5, pady=10)

    def set_button_states(self, can_go_prev, can_go_next):
        """Updates the state (enabled/disabled) of the navigation buttons."""
        self.prev_button.config(state=tk.NORMAL if can_go_prev else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if can_go_next else tk.DISABLED)