"""Non-learned green-screen ROI rule, based on original project's HSV foreground code."""
import cv2
import numpy as np


def foreground_crop(image, empty=False, min_area_ratio=.002, padding_ratio=.12, mode="original"):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("nonempty uint8 BGR image required")
    height, width = image.shape[:2]
    if height == 0 or width == 0 or min_area_ratio <= 0 or padding_ratio < 0:
        raise ValueError("positive dimensions/area threshold and nonnegative padding required")
    full = [0, 0, width, height]
    if empty:
        return full, {"method": "known_empty_full_frame", "foreground_found": False, "fallback": False}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mode not in {"original", "saturated_green"}:
        raise ValueError("unknown background rule")
    # Green-cast neutral objects are not a saturated green backdrop. This threshold
    # is fixed before classifier evaluation; it deliberately retains the pale tray.
    lower_saturation = 35 if mode == "original" else 180
    green = cv2.inRange(hsv, np.array([35, lower_saturation, 35], np.uint8), np.array([95, 255, 255], np.uint8))
    mask = cv2.bitwise_not(green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [record for record in stats[1:count] if record[4] / float(height * width) >= min_area_ratio]
    if not candidates:
        return full, {"method": "green_hsv_largest_component", "foreground_found": False,
            "fallback": True, "fallback_reason": "no component above fixed area threshold"}
    x, y, box_width, box_height, area = max(candidates, key=lambda record: record[4])
    padding_x = max(8, int(np.ceil(box_width * padding_ratio)))
    padding_y = max(8, int(np.ceil(box_height * padding_ratio)))
    x0, y0 = max(0, int(x) - padding_x), max(0, int(y) - padding_y)
    x1, y1 = min(width, int(x + box_width) + padding_x), min(height, int(y + box_height) + padding_y)
    return [x0, y0, x1 - x0, y1 - y0], {"method": "green_hsv_largest_component",
        "foreground_found": True, "fallback": False, "component_area": int(area),
        "component_box_xywh": list(map(int, [x, y, box_width, box_height]))}
