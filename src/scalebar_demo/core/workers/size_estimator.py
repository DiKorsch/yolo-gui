import cv2
import numpy as np
import typing as T

from scalebar.core.size import Size

from ultralytics.engine import results
from scalebar_demo.core.workers.scalebar import Result

class SizeEstimator:

    def __init__(self, size_per_square: float, size: Size, detector: str):
        from scalebar_demo.core.workers.detection import DetectionWorker
        from scalebar_demo.core.workers.scalebar import ScalebarProcessor

        self.scale_estimator = ScalebarProcessor(size_per_square, size=size)
        self.detector = DetectionWorker(detector)

    def __call__(self, img):

        preds = self.detector.predict(img)
        scbar = self.scale_estimator.estimate(img)
        is_seg = self.detector.is_seg
        scale = scbar.scale
        res = img.copy()

        if scale is None:
            if is_seg:
                return self.detector.show_seg(res, preds)
            else:
                return self.detector.show_boxes(res, preds)


        res = show_predictions(res, preds, scale, is_seg=is_seg)
        res = plot_corners(res, scbar, scale)
        return res, scale

def plot_corners(img: np.ndarray, scbar: Result, scale: float) -> np.ndarray:

    ys, xs = scbar.corners.transpose(1, 0)
    for x, y in zip(xs, ys):
        img = cv2.circle(img, (x + scbar.position.x, y + scbar.position.y), 2, (0, 0, 255), -1)

    return img

def show_predictions(img: np.ndarray, preds: T.List[results.Results], scale: float, *, is_seg: bool = False) -> np.ndarray:
    from scalebar_demo.core.workers.detection import putText

    for pred in preds:
        for i, box in enumerate(pred.boxes.cpu().numpy()):
            name = pred.names[int(box.cls)]

            if is_seg:
                coords = pred.masks.xy[i].astype(np.int32)
                x0, y0 = np.min(coords, axis=0)
                x1, y1 = np.max(coords, axis=0)
                area = cv2.contourArea(coords) / scale**2
                w, h = (x1 - x0) / scale, (y1 - y0) / scale
                cv2.polylines(img, [coords], True, (0, 255, 0), 2)
            else:
                coords = box.xyxy[0]
                x0, y0, x1, y1 = map(int, coords)
                area = (x1 - x0) * (y1 - y0) / scale**2
                w, h = (x1 - x0) / scale, (y1 - y0) / scale
                img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            name = f"{name} {w/10:.2f}x{h/10:.2f}cm ({area/100:.2f}cm^2)"
            putText(img, name, (x0, y0, x1, y1), color=(0, 0, 255))
    return img
