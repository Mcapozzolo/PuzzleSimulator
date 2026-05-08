import cv2
import numpy as np


class BlackFrameDetector:
    def __init__(self, min_area_ratio=0.05, max_area_ratio=0.95):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def order_points(self, pts):
        pts = np.array(pts, dtype=np.float32)

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def detect(self, image):
        """
        Detect the black rectangular frame in a top-down / warped image.

        Returns:
            frame_corners: np.ndarray shape (4,2) or None
            frame_bbox: (x, y, w, h) or None
            debug_image: RGB/BGR image with overlay
            mask: binary mask used for detection
        """
        debug = image.copy()
        h, w = image.shape[:2]
        img_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # black frame should stay dark
        _, mask = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, debug, mask

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * self.min_area_ratio:
                continue
            if area > img_area * self.max_area_ratio:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, ww, hh = cv2.boundingRect(cnt)
            rect_area = ww * hh
            if rect_area <= 0:
                continue

            fill_ratio = area / rect_area

            # Prefer 4-corner rectangles, but tolerate larger approximations slightly
            score = area
            if len(approx) == 4:
                score += 1_000_000
            score += fill_ratio * 10_000

            candidates.append((score, cnt, approx))

        if not candidates:
            return None, None, debug, mask

        candidates.sort(key=lambda t: t[0], reverse=True)
        _, best_cnt, best_approx = candidates[0]

        if len(best_approx) == 4:
            corners = best_approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(best_cnt)
            corners = cv2.boxPoints(rect)

        corners = self.order_points(corners)
        x, y, ww, hh = cv2.boundingRect(best_cnt)

        cv2.drawContours(debug, [corners.astype(np.int32)], -1, (0, 255, 0), 3)
        cv2.rectangle(debug, (x, y), (x + ww, y + hh), (255, 0, 0), 2)

        for i, (px, py) in enumerate(corners.astype(int)):
            cv2.circle(debug, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                debug,
                str(i),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return corners, (x, y, ww, hh), debug, mask

    def crop_inside_frame(self, image, frame_corners, margin_px=10):
        """
        Simple crop using axis-aligned bounding box of detected frame.
        Good when image is already warped to top-down.
        """
        x_coords = frame_corners[:, 0]
        y_coords = frame_corners[:, 1]

        x_min = max(0, int(np.min(x_coords)) + margin_px)
        y_min = max(0, int(np.min(y_coords)) + margin_px)
        x_max = min(image.shape[1], int(np.max(x_coords)) - margin_px)
        y_max = min(image.shape[0], int(np.max(y_coords)) - margin_px)

        if x_max <= x_min or y_max <= y_min:
            return None, None

        cropped = image[y_min:y_max, x_min:x_max].copy()
        return cropped, (x_min, y_min, x_max - x_min, y_max - y_min)