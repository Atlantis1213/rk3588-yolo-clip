# 固定 HSV 绿幕前景规则：输入 uint8 BGR，返回 [x,y,w,h] 与裁剪诊断记录。
# 不使用学习模型；empty 是调用方提供的已知状态，不是本函数预测的类别。

"""Non-learned green-screen ROI rule, based on original project's HSV foreground code."""
import cv2
import numpy as np


def foreground_crop(image, empty=False, min_area_ratio=0.002, padding_ratio=0.12, mode="original"):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("nonempty uint8 BGR image required")
    height, width = image.shape[:2]
    if height == 0 or width == 0 or min_area_ratio <= 0 or padding_ratio < 0:
        raise ValueError("positive dimensions/area threshold and nonnegative padding required")
    full = [0, 0, width, height]
    if empty:
        return full, {
            "method": "known_empty_full_frame",
            "foreground_found": False,
            "fallback": False,
        }
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mode not in {"original", "saturated_green"}:
        raise ValueError("unknown background rule")
    # Green-cast neutral objects are not a saturated green backdrop. This threshold
    # is fixed before classifier evaluation; it deliberately retains the pale tray.
    # original 保留旧阈值，saturated_green 只把高饱和绿当背景，降低浅绿偏色物体被抠除的概率。
    lower_saturation = 35 if mode == "original" else 180
    green = cv2.inRange(
        hsv, np.array([35, lower_saturation, 35], np.uint8), np.array([95, 255, 255], np.uint8)
    )
    # green 为背景掩码，取反后才是待保留前景；HSV 范围不是 RGB 三通道阈值。
    mask = cv2.bitwise_not(green)
    kernel = np.ones((5, 5), np.uint8)
    # 先开运算去小噪点，再闭运算填小孔，最后平滑并重新二值化后做连通域分析。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 跳过 stats[0] 的背景域；按整图面积比例过滤小连通域，阈值不由分类结果动态调整。
    candidates = [
        record for record in stats[1:count] if record[4] / float(height * width) >= min_area_ratio
    ]
    if not candidates:
        return full, {
            "method": "green_hsv_largest_component",
            "foreground_found": False,
            "fallback": True,
            "fallback_reason": "no component above fixed area threshold",
        }
    # 只选面积最大的合格连通域，不合并多个物体；找不到时显式记录 fallback 并返回整图。
    x, y, box_width, box_height, area = max(candidates, key=lambda record: record[4])
    # 按包围框比例扩边，至少保留 8 像素上下文，再截断到图像边界。
    padding_x = max(8, int(np.ceil(box_width * padding_ratio)))
    padding_y = max(8, int(np.ceil(box_height * padding_ratio)))
    x0, y0 = max(0, int(x) - padding_x), max(0, int(y) - padding_y)
    x1, y1 = min(width, int(x + box_width) + padding_x), min(
        height, int(y + box_height) + padding_y
    )
    return [x0, y0, x1 - x0, y1 - y0], {
        "method": "green_hsv_largest_component",
        "foreground_found": True,
        "fallback": False,
        "component_area": int(area),
        "component_box_xywh": list(map(int, [x, y, box_width, box_height])),
    }
