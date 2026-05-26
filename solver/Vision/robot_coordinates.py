from PIL import report


class RobotCoordinateMapper:
    def __init__(self, workspace_size_px, workspace_size_mm, crop_offset_px=(0, 0)):
        self.workspace_w_px, self.workspace_h_px = workspace_size_px
        self.workspace_w_mm, self.workspace_h_mm = workspace_size_mm
        self.crop_offset_x, self.crop_offset_y = crop_offset_px

    def crop_px_to_workspace_px(self, point_px):
        x, y = point_px
        return x + self.crop_offset_x, y + self.crop_offset_y

    def workspace_px_to_mm(self, point_px):
        x_px, y_px = point_px
        x_mm = (x_px / self.workspace_w_px) * self.workspace_w_mm
        y_mm = (y_px / self.workspace_h_px) * self.workspace_h_mm
        return round(x_mm, 2), round(y_mm, 2)

    def crop_px_to_mm(self, point_px):
        return self.workspace_px_to_mm(
            self.crop_px_to_workspace_px(point_px)
        )

    def transform_report_to_robot_command(self, report, pick_center=None):
        if pick_center is not None:
            pick_px = (pick_center["col"], pick_center["row"])
        else:
            pick_px = (report["y0"], report["x0"])

        place_px = (report["y1"], report["x1"])

        pick_mm = self.crop_px_to_mm(pick_px)
        place_mm = self.crop_px_to_mm(place_px)

        return {
            "piece_id": report["piece_id"],
            "pick_x_mm": pick_mm[0],
            "pick_y_mm": pick_mm[1],
            "place_x_mm": place_mm[0],
            "place_y_mm": place_mm[1],
            "rotation_deg": report["rotation_deg"],
        }