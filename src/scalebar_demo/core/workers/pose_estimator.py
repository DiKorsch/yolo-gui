import numpy as np
import cv2

from mmpose.apis import MMPoseInferencer

class PoseEstimator:
    def __init__(self, threed=False):
        if threed:
            self.model = MMPoseInferencer(pose3d="human3d")
        else:
            self.model = MMPoseInferencer(pose2d="human")

    def __call__(self, frame: np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        generator = self.model(frame, return_vis=True)
        result = next(generator)
        return result["visualization"][0]
