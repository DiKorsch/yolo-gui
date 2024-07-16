import cv2
import numpy as np
import typing as T

from scalebar.core.size import Size
from ultralytics.engine import results

class SizeEstimator:

    def __init__(self, size_per_square: float, size: Size, detector: str):
        from scalebar_demo.core.workers.detection import DetectionWorker
        from scalebar_demo.core.workers.scalebar import ScalebarProcessor

        self.scale_estimator = ScalebarProcessor(size_per_square, size=size)
        self.detector = DetectionWorker(detector)

    def __call__(self, img):
        from scalebar_demo.core.workers.detection import putText

        preds: T.List[results.Results] = self.detector.predict(img)
        scbar = self.scale_estimator.estimate(img)
        is_seg = self.detector.is_seg
        scale = scbar.scale
        if scale is None:
            if is_seg:
                return self.detector.show_seg(img, preds)
            else:
                return self.detector.show_boxes(img, preds)

        res = img.copy()
        for pred in preds:

            for i, box in enumerate(pred.boxes.cpu().numpy()):
                name = pred.names[int(box.cls)]

                if is_seg:
                    coords = pred.masks.xy[i].astype(np.int32)
                    x0, y0 = np.min(coords, axis=0)
                    x1, y1 = np.max(coords, axis=0)
                    area = cv2.contourArea(coords) / scale**2
                    w, h = (x1 - x0) / scale, (y1 - y0) / scale
                    cv2.polylines(res, [coords], True, (0, 255, 0), 2)
                else:
                    coords = box.xyxy[0]
                    x0, y0, x1, y1 = map(int, coords)
                    area = (x1 - x0) * (y1 - y0) / scale**2
                    w, h = (x1 - x0) / scale, (y1 - y0) / scale
                    res = cv2.rectangle(res, (x0, y0), (x1, y1), (0, 255, 0), 2)

                name = f"{name} {w/10:.2f}x{h/10:.2f}cm ({area/100:.2f}cm^2)"
                putText(res, name, (x0, y0, x1, y1), color=(0, 0, 255))
        return res, scale
