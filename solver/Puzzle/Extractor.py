import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from solver.Img.GreenScreen import remove_background
from solver.Img.filters import export_contours, export_contours_without_colormatching

PREPROCESS_DEBUG_MODE = 0


def show_image(img, ind=None, name="image", show=True):
    """Helper used for matplotlib image display"""
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()


def show_contours(contours, imgRef):
    """Helper used for matplotlib contours display"""
    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "cont.png"), whiteImg)

class Extractor:
    """
    Class used for preprocessing and pieces extraction
    """

    def __init__(self, path, viewer=None, green_screen=False, factor=0.84):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)

        if green_screen:
            # ...existing green screen code...
            self.img = cv2.medianBlur(self.img, 5)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print("Resizing with factor", divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), self.img)
            remove_background(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), factor=factor)
            self.img_bw = cv2.imread(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "green_background_removed.png"), cv2.IMREAD_GRAYSCALE
            )
        else:
            # Für schwarze Teile: Vorverarbeitung für einheitlichen Look
            self.img_bw = self.preprocess_black_pieces()

        self.viewer = viewer
        self.green_ = green_screen
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def log(self, *args):
        """Helper function to log informations to the GUI"""
        print(" ".join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def extract(self):
        """
        Perform the preprocessing of the image and call functions to extract
        informations of the pieces.
        """

        kernel = np.ones((3, 3), np.uint8)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"))

        ### Implementation of random functions, actual preprocessing is down below

        def fill_holes():
            """filling contours found (and thus potentially holes in pieces)"""

            contour, _ = cv2.findContours(
                self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        #def generated_preprocesing():
            #ret, self.img_bw = cv2.threshold(
                #self.img_bw, 254, 255, cv2.THRESH_BINARY_INV
            #)
            #cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "otsu_binarized.png"), self.img_bw)
            #self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            #self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        def generated_preprocesing():
            """Robust preprocessing for real puzzle images with crack-free contours."""

            import cv2, numpy as np, os

            # Grayscale conversion
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # Illumination normalization (removes uneven lighting)
            blur_bg = cv2.GaussianBlur(gray, (55, 55), 0)
            norm = cv2.addWeighted(gray, 1.5, blur_bg, -0.5, 0)
            norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

            # Gentle contrast boost
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)

            # Adaptive threshold with slightly smaller block size
            self.img_bw = cv2.adaptiveThreshold(
                gray_eq,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                5
            )

            # Add edge reinforcement
            edges = cv2.Canny(gray_eq, 20, 80)
            self.img_bw = cv2.bitwise_or(self.img_bw, edges)

            # Close cracks & fill gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            # Fill internal holes completely
            contours, _ = cv2.findContours(self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

            # Smooth & denoise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel_open, iterations=1)
            self.img_bw = cv2.GaussianBlur(self.img_bw, (3, 3), 0)

            # Optional: final threshold to clean up blur residues
            _, self.img_bw = cv2.threshold(self.img_bw, 127, 255, cv2.THRESH_BINARY)

            # Save intermediate debug images (optional)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "pre_gray_eq.png"), gray_eq)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "pre_norm.png"), norm)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "adaptive_fixed.png"), self.img_bw)

        def real_preprocessing():
            """Apply morphological operations on base image."""
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        ### PREPROCESSING: starts there

        # With this we apply morphologic operations (CLOSE, OPEN and GRADIENT)
        if not self.green_:
            generated_preprocesing()
            self.separate_pieces()
        else:
            real_preprocessing()
            self.separate_pieces()
        # These prints are activated only if the PREPROCESS_DEBUG_MODE variable at the top is set to 1
        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        # With this we fill the holes in every contours, to make sure there is no fragments inside the pieces
        #if not self.green_:
        #    fill_holes()

        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage(
                "Binarized treshold", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png")
            )

        contours, hier = cv2.findContours(
            self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Sicherheitsprüfung
        if not contours:
            self.log("No contours found")
            return None

        self.log("Found nb pieces (raw contours):", len(contours))

        # Filter tiny contours early
        min_area = 1000  # Mindestgröße in Pixeln
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not contours:
            self.log("No contours after size filtering")
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
            self.log("Found nb pieces after manual setting: " + str(len(contours)))
        else:
            # Try to remove useless contours
            if len(contours) < 2:
                self.log("Not enough contours found:", len(contours))
                return None

            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            max = contours[1].shape[0]
            contours = [elt for elt in contours if elt.shape[0] > max / 3]
            self.log("Found nb pieces after removing bad ones: " + str(len(contours)))

        if PREPROCESS_DEBUG_MODE == 1:
            show_contours(contours, self.img_bw)  # final contours

        ### PREPROCESSING: the end

        # In case with fail to find the pieces, we fill some holes and then try again
        # while True: # TODO Add this at the end of the project, it is a fallback tactic

        self.log(">>> START contour/corner detection")
        puzzle_pieces = export_contours_without_colormatching(
            self.img,
            self.img_bw,
            contours,
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "contours.png"),
            5,
            viewer=self.viewer,
            green=self.green_,
        )
        if puzzle_pieces is None:
            # Export contours error
            return None
        return puzzle_pieces

    def preprocess_black_pieces(self):
        """Specialized preprocessing for black puzzle pieces on light background"""
        import cv2
        import numpy as np
        import os

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

        # Debug output
        debug_dir = os.path.join("resources", "debug")
        os.makedirs(debug_dir, exist_ok=True)

        cv2.imwrite(os.path.join(debug_dir, "01_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, "02_binary.png"), binary)
        cv2.imwrite(os.path.join(debug_dir, "03_filled.png"), filled)
        cv2.imwrite(os.path.join(debug_dir, "04_separated.png"), separated)
        cv2.imwrite(os.path.join(debug_dir, "05_cleaned.png"), cleaned)
        cv2.imwrite(os.path.join(debug_dir, "06_result.png"), result)

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
            self.log(f"Not enough contours found: {len(contours)}")
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
            self.log(f"Not enough valid contours after filtering: {len(filtered_contours)}")
            return None

        self.log(f"Found {len(filtered_contours)} valid pieces")
        return filtered_contours
