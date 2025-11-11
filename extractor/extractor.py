from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from .piece import Piece, PieceTransformation

import abc


class Extractor(abc.ABC):
    """Abstract base class for puzzle piece extractors."""

    @abc.abstractmethod
    def extract_pieces_and_transformations(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> Tuple[List[Piece], List[PieceTransformation], Optional[List[np.ndarray]]]:
        """
        Main API: (image) -> (pieces[], transformations[], debug_images[])
        Executes the full pipeline sequentially.
        """
        pass


class MockExtractor(Extractor):
    """Pipeline-based puzzle piece extractor."""

    def __init__(self):
        self.step_descriptions_ = [
            "Input image loaded",
            "Grayscale conversion applied",
            "Contours detected",
            "Pieces detected",
            "Pieces extracted",
            "Pieces corrected",
            "Corners detected",
            "Puzzle assembled",
        ]

    def extract_pieces_and_transformations(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> Tuple[List[Piece], List[PieceTransformation], Optional[List[np.ndarray]]]:
        """
        Main API: (image) -> (pieces[], transformations[], debug_images[])
        Executes the full pipeline sequentially.
        """

        assets_path = f"{Path(__file__).parent.parent.resolve()}/assets"
        mock_images_path = Path(f"{assets_path}/mock_images")
        mock_images = []

        for p in sorted(mock_images_path.iterdir()):
            if p.is_file() and p.suffix.lower() == ".png":
                img = np.array(Image.open(p))
                mock_images.append(img)

        return [], [], mock_images
