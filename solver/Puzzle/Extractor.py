import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging as log

from solver.Img.filters import export_contours_without_colormatching

PREPROCESS_DEBUG_MODE = 0


def show_contours(contours, img_ref) -> np.ndarray:
    """Helper used for matplotlib contours display"""
    # create a white RGB image to draw colored contours on
    h, w = img_ref.shape[:2]
    whiteImg = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 4, maxLevel=1)

    return whiteImg

class Extractor:
    """
    Class used for preprocessing and pieces extraction
    """

    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self.debug_images_ = []

    def extract(self):
        """
        Perform the preprocessing of the image and call functions to extract
        informations of the pieces.
        """
        self.img_bw = self.preprocess_black_pieces()
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        kernel = np.ones((3, 3), np.uint8)

        self.debug_images_.append(self.img.copy())

        self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
        self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        self.separate_pieces()

        self.debug_images_.append(self.img_bw.copy())

        contours, hier = cv2.findContours(
            self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Sicherheitsprüfung
        if not contours:
            log.info("No contours found")
            return None

        log.info("Found nb pieces (raw contours):", len(contours))

        # Filter tiny contours early
        min_area = 1000  # Mindestgröße in Pixeln
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not contours:
            log.info("No contours after size filtering")
            return None

        # With this we can manually set the maximum number of pieces manually, or we try to guess their number
        # to guess it, we only keep the contours big enough
        nb_pieces = None

        # TEMPORARY TO AVOID DEBUG ORGINAL:
        if len(sys.argv) < 0:
            # Number of pieces specified by user
            nb_pieces = int(sys.argv[2])
            contours = sorted(
                np.array(contours), key=lambda x: x.shape[0], reverse=True
            )[:nb_pieces]
            log.info("Found nb pieces after manual setting: " + str(len(contours)))
        else:
            # Try to remove useless contours
            if len(contours) < 2:
                log.info("Not enough contours found:", len(contours))
                return None

            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            max = contours[1].shape[0]
            contours = [elt for elt in contours if elt.shape[0] > max / 3]
            log.info("Found nb pieces after removing bad ones: " + str(len(contours)))

        self.debug_images_.append(show_contours(contours, self.img_bw))  # final contours

        log.info(">>> START contour/corner detection")
        puzzle_pieces, _ = export_contours_without_colormatching(
            self.img,
            self.img_bw,
            contours,
            5,
        )

        if puzzle_pieces is None:
            # Export contours error
            return None

        return puzzle_pieces, self.debug_images_

    def preprocess_black_pieces(self):
        """Specialized preprocessing for black puzzle pieces on light background"""

        # 1. Grayscale & blur for noise reduction
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Simple thresholding - dark pieces are clearly distinct
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # 3. Remove border regions and noise
        h, w = binary.shape
        border = 30
        binary[0:border, :] = 0
        binary[-border:, :] = 0
        binary[:, 0:border] = 0
        binary[:, -border:] = 0

        # 4. Fill holes (screw holes in pieces)
        kernel_close = np.ones((15,15), np.uint8)
        filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 5. Separate touching pieces
        kernel_erode = np.ones((3,3), np.uint8)
        separated = cv2.erode(filled, kernel_erode, iterations=1)

        # 6. Remove small artifacts and smooth edges
        kernel_clean = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(separated, cv2.MORPH_OPEN, kernel_clean)

        # 7. Connected components filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)

        # Size limits based on expected piece size
        min_size = (h * w) * 0.02  # 2% of image area
        max_size = (h * w) * 0.2   # 20% of image area

        result = np.zeros_like(cleaned)
        valid_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_size < area < max_size:
                result[labels == i] = 255
                valid_count += 1

        print(f"Found {valid_count} valid puzzle pieces")
        return result

    def separate_pieces(self):
        """Modified piece separation with better contour filtering"""
        # Original preprocessing
        self.img_bw = self.preprocess_black_pieces()

        # Find contours
        contours, _ = cv2.findContours(
            self.img_bw,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 2:
            log.info(f"Not enough contours found: {len(contours)}")
            return None

        # Filter contours by area and complexity
        img_area = self.img_bw.shape[0] * self.img_bw.shape[1]
        min_area = img_area * 0.02
        max_area = img_area * 0.15

        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # Zusätzlich: Prüfe Konturkomplexität
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 100:  # Mindestumfang
                    filtered_contours.append(cnt)

        if len(filtered_contours) < 2:
            log.info(f"Not enough valid contours after filtering: {len(filtered_contours)}")
            return None

        log.info(f"Found {len(filtered_contours)} valid pieces")
        return filtered_contours
