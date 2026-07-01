import numpy as np

from .. import config
from .Enums import TypeEdge, Directions


class Edge:
    """
    Wrapper for edges.
    Contains shape, colors, type and position information of a puzzle edge.
    """

    def __init__(
        self,
        shape,
        color=None,
        edge_type=TypeEdge.HOLE,
        connected=False,
        direction=Directions.N,
    ):
        self.shape = shape
        self.shape_backup = np.copy(shape)
        self.color = color
        self.type = edge_type
        self.connected = connected
        self.direction = direction

    def backup_shape(self):
        self.shape_backup = np.copy(self.shape)

    def restore_backup_shape(self):
        self.shape = np.copy(self.shape_backup)

    def compute_offset_shape(self, offset_px, piece_centroid):
        """Return self.shape shifted outward by offset_px pixels.

        The direction is decided from the piece centroid, so all points of the
        edge are shifted away from the piece centre. SmallSolver and Distance use
        this to compare a tolerance band instead of the raw noisy contour.
        """
        pts = np.asarray(self.shape, dtype=np.float32)
        if len(pts) < 2 or offset_px == 0 or piece_centroid is None:
            return pts.copy()

        normals = np.zeros_like(pts, dtype=np.float32)
        for i in range(len(pts)):
            prev_pt = pts[i - 1] if i > 0 else pts[0]
            next_pt = pts[i + 1] if i < len(pts) - 1 else pts[-1]
            tangent = next_pt - prev_pt
            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                continue
            tangent = tangent / norm
            normals[i] = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        centroid = np.asarray(piece_centroid, dtype=np.float32)
        to_centroid = centroid - pts
        if float(np.sum(normals * to_centroid)) > 0:
            normals = -normals

        return pts + normals * offset_px

    def is_compatible(self, e2, relaxed=False):
        """Return True when two edges may be matched.

        BORDER edges never match. HEAD/HEAD and HOLE/HOLE are rejected by default,
        because they cannot physically interlock. UNDEFINED remains allowed as a
        fallback for imperfect edge classification.
        """
        if self.type == TypeEdge.BORDER or e2.type == TypeEdge.BORDER:
            return False
        if (
            config.FORBID_SAME_TYPE_MATCH
            and self.type == e2.type
            and self.type in (TypeEdge.HEAD, TypeEdge.HOLE)
        ):
            return False
        return True
