# Configuration used by the small/agglomerative solver integration.
# Keep this file small: it only contains values needed by Distance/Mover/Edge/SmallSolver.

DEBUG_FILE_OUTPUT = 1
DEBUG_SHOW_DIAGRAMS = 0
DEBUG_ALT_SOLVER = 0
DEBUG_PIECE_CENTERS = 0

# Shifts compared edge curves outward before matching. This makes the matcher less
# sensitive to small contour noise and manufacturing tolerance of the real parts.
EDGE_OFFSET = 12

# Max relative deviation from the corner-to-corner baseline to classify an edge as flat.
EDGE_FLAT_FRAC = 0.13

# Edge-match score weights.
MATCH_RESIDUAL_WEIGHT = 1.0
MATCH_CURVATURE_WEIGHT = 1.0

# Same type connectors cannot physically interlock.
FORBID_SAME_TYPE_MATCH = 1

# Optional guard for the old grid solver.
REQUIRE_BORDER_OUTWARD = 1
