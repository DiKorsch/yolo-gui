import numpy as np
import typing as T
import cv2

from ultralytics import YOLO
from ultralytics.engine import results

from scalebar_demo.core.workers.base import BaseWorker
from scalebar_demo.utils import putText

class Detector(BaseWorker):

    def __init__(self, weights: str):
        self.load_model(weights)

    def detect(self, frame: np.ndarray) -> T.List[results.Results]:
        return self.model.predict(source=frame, verbose=False)

    def load_model(self, weights: str):
        self.model = YOLO(weights)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        preds = self.detect(frame)

        for pred in preds:
            for box in pred.boxes.cpu().numpy():
                class_id = int(box.cls)
                name = pred.names[class_id]
                for coords in box.xyxy:
                    x0, y0, x1, y1 = map(int, coords)
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    putText(frame, name, (x0, y0, x1, y1))
        return frame
