from typing import List, Optional, Tuple
from pathlib import Path
from .piece import Piece, PieceTransformation, Image

import abc


class Extractor(abc.ABC):
    """Abstract base class for puzzle piece extractors."""

    @abc.abstractmethod
    def extract_pieces_and_transformations(
        self,
        image: Image,
        debug: bool = False,
    ) -> Tuple[List[Piece], List[PieceTransformation], Optional[List[Image]]]:
        """
        Main API: (image) -> (pieces[], transformations[], debug_images[])
        Executes the full pipeline sequentially.
        """
        pass


class MockExtractor(Extractor):
    """Pipeline-based puzzle piece extractor."""

    def extract_pieces_and_transformations(
        self,
        image: Image,
        debug: bool = False,
    ) -> Tuple[List[Piece], List[PieceTransformation], Optional[List[Image]]]:
        """
        Main API: (image) -> (pieces[], transformations[], debug_images[])
        Executes the full pipeline sequentially.
        """

        mock_images_path = Path("../assets/mock_images")
        mock_images = []

        for p in sorted(mock_images_path.iterdir()):
            if p.is_file() and p.suffix.lower() == ".png":
                img = Image.open(p)
                mock_images.append(img)

        return [], [], mock_images
