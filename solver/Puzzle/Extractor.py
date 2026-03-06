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
        """Specialized preprocessing for black puzzle pieces on light background,
        robust against soft shadows."""

        img = self.img.copy()
        h, w = img.shape[:2]

        # 1) Gray + HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 2) Illumination correction / shadow normalization
        # Large blur estimates the background lighting field.
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
        bg = np.clip(bg, 1, 255).astype(np.float32)
        gray_f = gray.astype(np.float32)

        # Normalize local brightness: makes shadows flatter
        norm = (gray_f / bg) * 180.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # 3) Slight blur after normalization
        norm_blur = cv2.GaussianBlur(norm, (5, 5), 0)
        hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 4) Threshold masks
        # Gray-based on normalized image
        _, mask_gray = cv2.threshold(norm_blur, 105, 255, cv2.THRESH_BINARY_INV)

        # HSV-based black mask
        h_chan, s_chan, v_chan = cv2.split(hsv_blur)
        mask_v = cv2.inRange(v_chan, 0, 85)
        mask_s = cv2.inRange(s_chan, 0, 170)

        mask_hsv = cv2.bitwise_and(mask_v, mask_s)

        # 5) Combine masks
        # Use OR instead of strict AND so shadowed real pieces are not lost.
        combined = cv2.bitwise_and(mask_gray, mask_hsv)

        # 6) Remove borders
        border = max(10, int(min(h, w) * 0.01))
        combined[0:border, :] = 0
        combined[-border:, :] = 0
        combined[:, 0:border] = 0
        combined[:, -border:] = 0

        # 7) Morphology
        kernel_close = np.ones((13, 13), np.uint8)
        filled = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open)

        # 8) Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)

        component_areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 400:
                component_areas.append(area)

        result = np.zeros_like(cleaned)

        if len(component_areas) == 0:
            print("Found 0 valid puzzle pieces")
            return result

        median_area = float(np.median(component_areas))

        min_size = median_area * 0.35
        max_size = median_area * 2.50

        valid_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_size <= area <= max_size:
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
