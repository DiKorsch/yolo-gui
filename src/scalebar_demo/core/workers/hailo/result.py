import numpy as np
import cv2
import typing as T
import time
import structlog

from dataclasses import dataclass

try:
    from hailo_model_zoo.core.datasets.datasets_info import CLASS_NAMES_COCO # type: ignore
except ImportError:
    pass

from scalebar_demo.core.workers.hailo.postprocessing import postproccess

logger = structlog.get_logger()

COLORS = [ [int(c) for c in np.random.randint(0, 255, size=3)] for _ in CLASS_NAMES_COCO ]



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
        return [Result(**res) for res in postproccess(outputs, **kwargs)]

    def plot(self, image: np.ndarray, *,
            thresh: float = 0.5,
            alpha: float = 0.5,
            path: T.Optional[str] = None):
        global COLORS

        out = image.copy()

        keep = self.detection_scores > thresh
        boxes = self.detection_boxes
        masks = self.mask
        classes = self.detection_classes
        scores = self.detection_scores

        for idx in np.where(keep)[0]:
            mask = masks[idx]

            if not np.sum(mask):
                continue
            color = COLORS[idx % len(COLORS)]

            xmin, xmax = (boxes[idx, 0::2] * out.shape[1]).astype(np.int32)
            ymin, ymax = (boxes[idx, 1::2] * out.shape[0]).astype(np.int32)

            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)

            # t0 = time.time()
            mask = cv2.resize(mask, out.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            polygons, _ = mask_to_polygons(mask)
            # contour_t += time.time() - t0

            # t0 = time.time()
            if alpha is not None:
                mask = mask[:, :, np.newaxis] * np.array(color)
                cv2.addWeighted(mask.astype(out.dtype), alpha, out, 1, 0, out)
            # blend_t += time.time() - t0

            # t0 = time.time()
            for polygon in polygons:
                pol = [polygon.reshape((-1, 1, 2)).astype(np.int32)]
                out = cv2.polylines(out, pol, isClosed=True, color=color, thickness=2)
            # polylines_t += time.time() - t0

            # Draw text labels
            label = f"{CLASS_NAMES_COCO[int(classes[idx])]}"
            score = f"{100*scores[idx]:.0f}"
            text = f"{label}: {score}%"
            (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
            org = (xmin, ymin)
            # Rectangle for label background
            deltaY = max(h - ymin, 0)
            deltaX = max(-xmin, 0)
            out[
                np.max([ymin - h, 0]) : ymin + h // 2 + deltaY,
                np.max([xmin, 0]) : xmin + w + deltaX,
                :
            ] = color

            out = cv2.putText(
                out,
                text,
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=[255, 255, 255],
                thickness=2,
                lineType=cv2.FILLED,
            )

        # logger.info(f"[hailo plot] polylines: {polylines_t:.3f} | blend: {blend_t:.3f} | contour: {contour_t:.3f}")
        return out

def mask_to_polygons(mask, threshold: float = 0.5):
    # mask = np.ascontiguousarray(mask)
    contours, hierarchy = cv2.findContours((mask >= threshold).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = [x.flatten() + 0.5 for x in contours if len(x) >= 6]
    return res, has_holes
