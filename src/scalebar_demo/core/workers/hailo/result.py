import numpy as np
import cv2
import typing as T
import time
import structlog

from dataclasses import dataclass

from scalebar_demo.core.workers.hailo import postprocessing as pp

logger = structlog.get_logger()

COLORS = None


def unpack(arrs: T.List[np.ndarray], n: int, *, axis: int = 1) -> np.ndarray:
    res = [arr.reshape(-1, arr.shape[1] * arr.shape[2], n) for arr in arrs]
    return np.concatenate(res, axis=axis)

@dataclass
class Result:
    detection_boxes: np.ndarray
    detection_scores: np.ndarray
    detection_classes: np.ndarray
    mask: np.ndarray

    class_names: T.Optional[T.List[str]] = None

    @classmethod
    def postprocess(cls, raw_detections, prefix: str = "", *,
                    anchors: dict,
                    img_dims: T.List[int],
                    classes: int = 80,
                    class_names: T.Optional[T.List[str]] = None,
                    score_threshold: float = 0.5,
                    nms_iou_thresh: float = 0.5,
                ) -> T.List["Result"]:
        global COLORS
        assert class_names is None or len(class_names) == classes, \
            f"Class names length {len(class_names)} != classes {classes}"
        if COLORS is None:
            # lazy init colors
            COLORS = [ [int(c) for c in np.random.randint(0, 255, size=3)] for _ in range(classes) ]

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

        detections = [raw_detections[name] for name in order]

        num_classes = classes
        image_dims = tuple(img_dims)
        strides = anchors["strides"]
        reg_max = anchors["regression_length"]
        score_thres = score_threshold
        iou_thres = nms_iou_thresh

        t0 = time.time()
        decoded_boxes = pp.decode(detections[:7:3], strides, image_dims, reg_max)
        dec_t = time.time() - t0

        # unpack data
        t0 = time.time()
        proto_data = detections[9]
        n_masks = proto_data.shape[-1]

        scores = unpack(detections[1:8:3], num_classes)
        coeffs = unpack(detections[2:9:3], n_masks)

        fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1), dtype=np.float32)
        scores_obj = np.concatenate([fake_objectness, scores], axis=-1)
        # print("scores", scores_obj.dtype)

        # re-arrange predictions for nms
        predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
        unpack_t = time.time() - t0
        # print("preds", predictions.dtype)

        t0 = time.time()
        nms_results = pp.non_max_suppression(predictions,
                                             conf_thres=score_thres,
                                             iou_thres=iou_thres,
                                             multi_label=True)
        nms_t = time.time() - t0
        res = []
        image_size = np.tile(image_dims, 2)
        proc_t = 0

        # for key, arr in nms_res[0].items():
        #     print(key, arr.shape, arr.dtype)

        for protos, nms_res in zip(proto_data, nms_results):
            t0 = time.time()
            masks = pp.process_mask(protos, nms_res["mask"], nms_res["detection_boxes"], image_dims)
            proc_t += time.time() - t0

            res.append(cls(
                mask=masks if masks is not None else [],
                detection_scores=nms_res["detection_scores"],
                detection_classes=nms_res["detection_classes"].astype(np.int32),
                detection_boxes=(nms_res["detection_boxes"] / image_size).astype(np.float32),
                class_names=class_names,
            ))

        logger.debug(f"[hailo - postprocess times] unpack: {unpack_t:.3f}s | nms: {nms_t:.3f}s | decode: {dec_t:.3f}s | mask process: {proc_t:.3f}s")
        return res

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
            polygons, _ = pp.mask_to_polygons(mask)
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

            if self.class_names is None:
                continue

            # Draw text labels
            label = f"{self.class_names[int(classes[idx])]}"
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

        return out
