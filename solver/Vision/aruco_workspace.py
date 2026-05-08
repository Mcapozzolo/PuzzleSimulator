import cv2
import numpy as np


class ArucoWorkspace:
    def __init__(
        self,
        marker_length_mm=40.0,
        output_size_px=(1200, 800),
        dictionary_name=cv2.aruco.DICT_4X4_50,
        required_ids=(0, 1, 2, 3),
    ):
        self.marker_length_mm = marker_length_mm
        self.output_size_px = output_size_px
        self.required_ids = tuple(required_ids)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)
        self.detector_params = cv2.aruco.DetectorParameters()

        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 53
        self.detector_params.adaptiveThreshWinSizeStep = 4
        self.detector_params.minMarkerPerimeterRate = 0.01
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 5
        self.detector_params.cornerRefinementMaxIterations = 50
        self.detector_params.cornerRefinementMinAccuracy = 0.01
        
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect_markers(self, image):
        corners, ids, rejected = self.detector.detectMarkers(image)

        if ids is None or len(ids) == 0:
            return {}, corners, ids

        ids = ids.flatten()
        marker_map = {}

        for marker_id, marker_corners in zip(ids, corners):
            pts = marker_corners.reshape((4, 2)).astype(np.float32)
            marker_map[int(marker_id)] = pts

        return marker_map, corners, ids

    def _marker_center(self, pts):
        return np.mean(pts, axis=0)

    def get_workspace_corners_from_markers(self, marker_map):
        """
        Robust mode:
        Use any 4 detected markers and sort them by their image position:
        top-left, top-right, bottom-right, bottom-left.

        This avoids hard dependency on marker IDs 0,1,2,3.
        """
        if len(marker_map) < 4:
            found = sorted(marker_map.keys())
            raise ValueError(f"Need 4 ArUco markers, but found only {len(marker_map)}: {found}")

        centers = []
        for marker_id, pts in marker_map.items():
            center = self._marker_center(pts)
            centers.append((marker_id, center))

        # sort by y coordinate: top row first, bottom row second
        centers_sorted_y = sorted(centers, key=lambda x: x[1][1])

        top = centers_sorted_y[:2]
        bottom = centers_sorted_y[2:4]

        # sort each row by x coordinate
        top = sorted(top, key=lambda x: x[1][0])
        bottom = sorted(bottom, key=lambda x: x[1][0])

        tl = top[0][1]
        tr = top[1][1]
        br = bottom[1][1]
        bl = bottom[0][1]

        print(f"[Aruco] Found marker IDs: {[m[0] for m in centers]}")
        print(f"[Aruco] Auto-sorted workspace markers:")
        print(f"  TL: id={top[0][0]}")
        print(f"  TR: id={top[1][0]}")
        print(f"  BR: id={bottom[1][0]}")
        print(f"  BL: id={bottom[0][0]}")

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def warp_workspace(self, image, workspace_corners):
        out_w, out_h = self.output_size_px

        dst = np.array(
            [
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1],
            ],
            dtype=np.float32,
        )

        H = cv2.getPerspectiveTransform(workspace_corners, dst)
        warped = cv2.warpPerspective(image, H, (out_w, out_h))
        return warped, H

    def pixel_to_workspace_coords(self, point_px, workspace_mm_size):
        """
        Convert a point from warped image pixel coordinates to mm coordinates.
        workspace_mm_size = (width_mm, height_mm)
        """
        out_w, out_h = self.output_size_px
        width_mm, height_mm = workspace_mm_size

        x_px, y_px = point_px
        x_mm = (x_px / out_w) * width_mm
        y_mm = (y_px / out_h) * height_mm
        return float(x_mm), float(y_mm)

    def draw_detected_markers(self, image, corners, ids):
        vis = image.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        return vis

    def process(self, image):
        """
        Full pipeline:
        - detect markers
        - compute workspace corners
        - warp workspace
        Returns:
            warped_image, homography, debug_image, workspace_corners
        """
        marker_map, corners, ids = self.detect_markers(image)
        debug_image = self.draw_detected_markers(image, corners, ids)

        try:
            workspace_corners = self.get_workspace_corners_from_markers(marker_map)
        except ValueError as e:
            cv2.imwrite("assets/DEBUG_ROBOT_RUN/aruco_failed_debug.png", debug_image)
            print(f"[Aruco] Detection failed. Debug saved: assets/DEBUG_ROBOT_RUN/aruco_failed_debug.png")
            raise e
        warped, H = self.warp_workspace(image, workspace_corners)

        # draw polygon on debug image
        pts = workspace_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        return warped, H, debug_image, workspace_corners