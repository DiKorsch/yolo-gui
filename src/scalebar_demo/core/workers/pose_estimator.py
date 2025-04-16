import numpy as np
import cv2
import typing as T

from ultralytics.engine import results
from scalebar_demo.core.models import Purpose

from itertools import product

class PoseEstimator:
    available_snapshots = [
        f"yolo{ver}{size}{task}.pt"
        for ver, size, task in product(["11"], ["n", "s", "m", "l", "x"], ["-pose"])
    ]
    def __init__(self, threed=False, weights: str = "yolo11n-pose.pt", *, ncnn: bool = True):

        assert weights in PoseEstimator.available_snapshots, \
            f"Snapshot {weights} not available: {PoseEstimator.available_snapshots}"

        self.threed = threed
        self.model = Purpose.pose_estimation(weights, ncnn=ncnn, threed=threed)

    def __call__(self, frame: np.ndarray):
        if self.threed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            generator = self.model(frame, return_vis=True)
            result = next(generator)
            vis = result["visualization"][0]
            _, w, _ = frame.shape
            _, vis3d = vis[:, :w], vis[:, w:]
            return vis3d
        else:
            preds: T.List[results.Results] = self.model(frame, verbose=False)
            return preds[0].plot(boxes=False)

            # generator = self.model(frame, return_vis=True)
            # result = next(generator)
            # return result["visualization"][0]
