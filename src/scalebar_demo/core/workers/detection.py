import cv2
import numpy as np
import typing as T

try:
    from ultralytics import YOLO
    from ultralytics.engine import results
    HAS_YOLO = True
except ImportError:
    import warnings
    warnings.warn("Ultralytics YOLO is not available. Please install it.")
    HAS_YOLO = False
from itertools import product

from scalebar_demo.utils import putText

class DetectionWorker:
    available_snapshots = [
        f"yolo{ver}{size}{task}.pt"
        for ver, size, task in product(["v8", "11"], ["n", "s", "m", "l", "x"], ["", "-seg"])
    ]

    def __init__(self, snapshot: str):
        assert HAS_YOLO, "Ultralytics YOLO is not available. Please install it."
        assert snapshot in DetectionWorker.available_snapshots, \
            f"Snapshot {snapshot} not available: {DetectionWorker.available_snapshots}"
        self.yolo = YOLO(snapshot)
        self.is_seg = "seg" in snapshot

    def show_boxes(self, image: np.ndarray, preds: T.List["results.Results"]) -> np.ndarray:
        for pred in preds:
            for box in pred.boxes.cpu().numpy():
                class_id = int(box.cls)
                name = pred.names[class_id]
                for coords in box.xyxy:
                    x0, y0, x1, y1 = map(int, coords)
                    image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    putText(image, name, (x0, y0, x1, y1))
        return image

    def show_seg(self, image: np.ndarray, preds: T.List["results.Results"]) -> np.ndarray:
        for pred in preds:
            masks = pred.masks
            if masks is None:
                continue
            names = [pred.names[int(box.cls)] for box in pred.boxes]
            xys = [xy.astype(np.int32) for xy in masks.xy]
            cv2.polylines(image, xys, True, (0, 255, 0), 2)

            for name, xy in zip(names, xys):
                x0, y0 = np.min(xy, axis=0)
                x1, y1 = np.max(xy, axis=0)
                putText(image, name, (x0, y0, x1, y1))
                # cv2.putText(image, name, np.mean(xy, axis=-1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # for xys in mask.xy:
            # image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        return image

    def predict(self, image: np.ndarray) -> T.List["results.Results"]:
        return self.yolo.predict(source=image, verbose=False)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        preds: T.List[results.Results] = self.predict(image)
        res = image.copy()
        if self.is_seg:
            return self.show_seg(res, preds)
        else:
            return self.show_boxes(res, preds)
