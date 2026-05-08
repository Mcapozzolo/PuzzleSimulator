import cv2
import numpy as np

from .aruco_workspace import ArucoWorkspace
from .frame_detection import BlackFrameDetector


class VisionPipeline:
    def __init__(
        self,
        marker_length_mm=40.0,
        workspace_output_size_px=(1200, 800),
        workspace_mm_size=(400.0, 300.0),
        aruco_ids=(0, 1, 2, 3),
        frame_margin_px=10,
    ):
        self.marker_length_mm = marker_length_mm
        self.workspace_output_size_px = workspace_output_size_px
        self.workspace_mm_size = workspace_mm_size
        self.aruco_ids = tuple(aruco_ids)
        self.frame_margin_px = frame_margin_px

        self.aruco = ArucoWorkspace(
            marker_length_mm=self.marker_length_mm,
            output_size_px=self.workspace_output_size_px,
            required_ids=self.aruco_ids,
        )

        self.frame_detector = BlackFrameDetector()

    def process_image(self, image):
        """
        Full vision preprocessing pipeline.

        Returns a dict with:
            {
                "original": ...,
                "aruco_debug": ...,
                "warped_workspace": ...,
                "frame_debug": ...,
                "frame_mask": ...,
                "cropped_frame": ...,
                "homography": ...,
                "workspace_corners": ...,
                "frame_corners": ...,
                "frame_bbox": ...,
                "cropped_bbox": ...,
            }
        """
        if image is None:
            raise ValueError("Input image is None.")

        original = image.copy()

        # 1) ArUco detection + workspace warp
        warped, H, aruco_debug, workspace_corners = self.aruco.process(original)

        # 2) Black frame detection on warped image
        frame_corners, frame_bbox, frame_debug, frame_mask = self.frame_detector.detect(
            warped
        )

        cropped_frame = None
        cropped_bbox = None

        # 3) Crop inside detected frame
        if frame_corners is not None:
            cropped_frame, cropped_bbox = self.frame_detector.crop_inside_frame(
                warped,
                frame_corners,
                margin_px=self.frame_margin_px,
            )

        return {
            "original": original,
            "aruco_debug": aruco_debug,
            "warped_workspace": warped,
            "frame_debug": frame_debug,
            "frame_mask": frame_mask,
            "cropped_frame": cropped_frame,
            "homography": H,
            "workspace_corners": workspace_corners,
            "frame_corners": frame_corners,
            "frame_bbox": frame_bbox,
            "cropped_bbox": cropped_bbox,
        }

    def process_image_from_path(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        return self.process_image(image)

    def pixel_to_workspace_mm(self, point_px):
        """
        Convert a point in the warped workspace image to mm.
        """
        return self.aruco.pixel_to_workspace_coords(
            point_px,
            self.workspace_mm_size,
        )

    def crop_pixel_to_workspace_mm(self, point_px_in_crop, cropped_bbox):
        """
        Convert a point from cropped-frame pixel coordinates
        into workspace mm coordinates.

        point_px_in_crop: (x, y) inside cropped image
        cropped_bbox: (x, y, w, h) of crop in warped workspace
        """
        if cropped_bbox is None:
            raise ValueError("cropped_bbox is None")

        crop_x, crop_y, _, _ = cropped_bbox
        local_x, local_y = point_px_in_crop

        global_x = crop_x + local_x
        global_y = crop_y + local_y

        return self.pixel_to_workspace_mm((global_x, global_y))

    def make_debug_collage(self, result, scale=0.5):
        """
        Build a simple 2x2 debug collage from the pipeline outputs.
        Useful for quick inspection.
        """
        imgs = []

        keys = ["original", "aruco_debug", "warped_workspace", "frame_debug"]
        for key in keys:
            img = result.get(key)
            if img is None:
                img = np.zeros((200, 300, 3), dtype=np.uint8)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgs.append(img)

        # Resize all to same size
        target_h = min(img.shape[0] for img in imgs)
        target_w = min(img.shape[1] for img in imgs)

        resized = [
            cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for img in imgs
        ]

        top = np.hstack([resized[0], resized[1]])
        bottom = np.hstack([resized[2], resized[3]])
        collage = np.vstack([top, bottom])

        if scale != 1.0:
            collage = cv2.resize(
                collage,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        return collage