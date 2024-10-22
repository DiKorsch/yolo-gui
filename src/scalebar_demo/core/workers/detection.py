# import cv2
import numpy as np
import typing as T

from ultralytics import YOLO
from ultralytics.engine import results
from itertools import product

from scalebar_demo.core.workers.base import BaseWorker

class DetectionWorker(BaseWorker):
    available_snapshots = [
        f"yolo{ver}{size}{task}.pt"
        for ver, size, task in product(["v8", "11"], ["n", "s", "m", "l", "x"], ["", "-seg"])
    ]

    def __init__(self, snapshot: str, *, extra_snapshots: T.List[str] = []):
        snapshots = DetectionWorker.available_snapshots + extra_snapshots
        assert snapshot in snapshots, \
            f"Snapshot {snapshot} not available: {snapshots}"
        self.model = YOLO(snapshot)
        if hasattr(self.model, "compile"):
            self.model.compile()
        self.is_seg = "seg" in snapshot

    def predict(self, image: np.ndarray) -> T.List[results.Results]:
        return self.model.predict(source=image, verbose=False)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        preds: T.List[results.Results] = self.predict(image)
        res = image.copy()
        return preds[0].plot(img=res)
