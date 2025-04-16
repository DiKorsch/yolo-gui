import numpy as np
import structlog
import re
import cv2
import typing as T

from dataclasses import dataclass
from scalebar_demo.core.workers.base import BaseWorker

logger = structlog.get_logger()

try:
    import hailo_platform as HAILO

    from hailo_model_zoo.core.datasets.datasets_info import CLASS_NAMES_COCO
    from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import yolov8_seg_postprocess
    from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import mask_to_polygons
except ImportError:
    # import warnings
    # warnings.warn("Hailo platform is not available. Please install hailo_platform and hailo model zoo.")
    HAILO_AVAILABLE = False
else:
    HAILO_AVAILABLE = True

COLORS = [ [int(c) for c in np.random.randint(0, 255, size=3)] for _ in range(80) ]



@dataclass
class Result:
    detection_boxes: np.ndarray
    detection_scores: np.ndarray
    detection_classes: np.ndarray
    mask: np.ndarray

    @classmethod
    def postprocess(cls, raw_detections, prefix: str = "", **kwargs) -> T.List["Result"]:

        order = [
            f"{prefix}/conv44", # shape: [80, 80, 64]
            f"{prefix}/conv45", # shape: [80, 80, 80]
            f"{prefix}/conv46", # shape: [80, 80, 32]
            f"{prefix}/conv60", # shape: [40, 40, 64]
            f"{prefix}/conv61", # shape: [40, 40, 80]
            f"{prefix}/conv62", # shape: [40, 40, 32]
            f"{prefix}/conv73", # shape: [20, 20, 64]
            f"{prefix}/conv74", # shape: [20, 20, 80]
            f"{prefix}/conv75", # shape: [20, 20, 32]
            f"{prefix}/conv48", # shape: [160, 160, 32]
        ]

        outputs = [raw_detections[name] for name in order]
        return [Result(**res) for res in yolov8_seg_postprocess(outputs, **kwargs)]

    def plot(self, image: np.ndarray, *,
            thresh: float = 0.5,
            alpha: float = 0.5,
            path: T.Optional[str] = None):
        global COLORS

        out = image.copy()

        keep = self.detection_scores > thresh
        boxes = self.detection_boxes[keep]
        masks = self.mask[keep]
        classes = self.detection_classes[keep]
        scores = self.detection_scores[keep]

        boxes[:, 0::2] *= out.shape[1]
        boxes[:, 1::2] *= out.shape[0]

        for idx, mask in enumerate(masks):
            if not np.sum(mask):
                continue
            xmin, ymin, xmax, ymax = boxes[idx].astype(np.int32)
            color = COLORS[idx % len(COLORS)]
            # color = [int(c) for c in np.random.randint(low=0, high=255, size=3, dtype=np.uint8)]
            polygons, _ = mask_to_polygons(mask)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color

            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)
            mask = cv2.resize(mask, out.shape[1::-1], interpolation=cv2.INTER_LINEAR).astype(out.dtype)
            out = cv2.addWeighted(mask, alpha, out, 1, 0)

            for polygon in polygons:
                pol = [polygon.reshape((-1, 1, 2)).astype(np.int32)]
                out = cv2.polylines(out, pol, isClosed=True, color=color, thickness=1)

            label = f"{CLASS_NAMES_COCO[int(classes[idx])]}"
            score = f"{100*scores[idx]:.0f}"
            text = f"{label}: {score}%"
            (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
            org = (xmin, ymin)
            # Rectangle for label background
            deltaY = max(h - org[1], 0)
            deltaX = max(-org[0], 0)
            out[
                np.max([org[1] - h, 0]) : org[1] + h // 2 + deltaY, np.max([org[0], 0]) : org[0] + w + deltaX, :
            ] = color

            out = cv2.putText(
                out,
                text,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=[255, 255, 255],
                thickness=2,
                lineType=cv2.FILLED,
            )

        return out



class HailoWorker(BaseWorker):
    target = None

    def __init__(self, hef_path: str, conf: dict):
        assert HAILO_AVAILABLE, "Hailo platform is not available. Please install hailo_platform."
        devices = HAILO.Device.scan()

        self.target = HAILO.VDevice(device_ids=devices)

        hef = HAILO.HEF(hef_path)
        inputs = hef.get_input_vstream_infos()
        assert len(inputs) == 1, "Only one input is supported"
        input_info = self.input_info = inputs[0]
        self.prefix = re.match(r"([a-zA-Z0-9_]+)/.*", input_info.name).group(1)

        logger.info(f"[hailo] Input name: {input_info.name} | shape: {input_info.shape}")
        logger.info(f"[hailo] Estimated model prefix: {self.prefix}")

        config_params = HAILO.ConfigureParams.create_from_hef(hef, interface=HAILO.HailoStreamInterface.PCIe)
        self.netgroup = self.target.configure(hef, config_params)[0]
        self.netgroup_params = self.netgroup.create_params()

        self.conf = conf

    def __call__(self, image: np.ndarray) -> np.ndarray:

        _params = dict(configured_network=self.netgroup, quantized=False, format_type=HAILO.FormatType.FLOAT32)
        input_vstreams_params = HAILO.InputVStreamParams.make_from_network_group(**_params)
        output_vstreams_params = HAILO.OutputVStreamParams.make_from_network_group(**_params)
        X = np.expand_dims(cv2.resize(image, (640, 640)), axis=0).astype(np.float32)
        with HAILO.InferVStreams(self.netgroup, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with self.netgroup.activate(self.netgroup_params):
                raw_detections = infer_pipeline.infer({self.input_info.name: X})
                result = Result.postprocess(raw_detections, prefix=self.prefix, **self.conf)[0]

        return result.plot(image, thresh=self.conf["score_threshold"], alpha=0.5)

    def __del__(self):
        if self.target:
            self.target.release()
            self.target = None
