import numpy as np
from typing import List, Tuple

Point = Tuple[int, int]


class Edge:
    """Represents one side between two corners."""
    def __init__(self, start: Point, end: Point, contour_segment: np.ndarray, length: int, kind: str):
        self.start = start
        self.end = end
        self.contour_segment = contour_segment  # Nx2 array of contour points
        self.kind = kind                        # 'flat' | 'innie' | 'outie'
        self.length = length


class Piece:
    """Represents one puzzle piece with local geometry."""
    def __init__(self, piece_img: np.ndarray, corners: List[Point], edges: List[Edge]):
        self.piece_img = piece_img
        self.corners = corners
        self.edges = edges


class PieceTransformation:
    """Global transform of a piece in scene coordinates."""
    def __init__(self, x: float, y: float, rotation_deg: float):
        self.x = x
        self.y = y
        self.rotation_deg = rotation_deg