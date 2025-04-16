import enum
import torch as th

from ultralytics import YOLO

class Purpose(enum.Enum):
    pose_estimation = enum.auto()
    detection = enum.auto()


    def __call__(self, weights: str, *, ncnn: bool = True, **kwargs) -> "th.nn.Module":
        if self == Purpose.pose_estimation:
            if kwargs.get("threed", False):
                try:
                    from mmpose.apis import MMPoseInferencer
                except ImportError:
                    raise ImportError("MMPose is not available. Please install it.")

                return MMPoseInferencer(pose3d="human3d")

            else:
                return self.load_yolo(weights, ncnn=ncnn)

    def load_yolo(self, weights: str, *, ncnn: bool) -> "th.nn.Module":
        assert YOLO is not None, "YOLO is not available. Please install it."
        assert th is not None, "Torch is not available. Please install it."
        model = YOLO(weights)
        if ncnn:
            model = YOLO(model.export(format="ncnn"))
        return model
