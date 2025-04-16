
import typing as T
import numpy as np
import structlog
import time
import cv2

try:
    from hailo_model_zoo.core.postprocessing import instance_segmentation_postprocessing as seg_post # type: ignore
except ImportError:
    pass

logger = structlog.get_logger()

def _unpack(arrs: T.List[np.ndarray], n: int, *, axis: int = 1) -> np.ndarray:
    try:
        res = [arr.reshape(-1, arr.shape[1] * arr.shape[2], n) for arr in arrs]
    except ValueError:
        reshapes = [f"{arr.shape} -> {(-1, arr.shape[1] * arr.shape[2], n)}" for arr in arrs]
        logger.error(f"could not reshape arrays: {reshapes}")
        raise
    return np.concatenate(res, axis=axis)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _softmax(x, axis=-1):
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=-1, keepdims=True)

DECODE_CACHE = {}
def _get_decode_cache(w, h, stride, dtype=np.float32):
    global DECODE_CACHE

    key = (w, h, stride)
    if key not in DECODE_CACHE:
        grid_x = np.arange(w, dtype=dtype) + 0.5
        grid_y = np.arange(h, dtype=dtype) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
        DECODE_CACHE[key] = center
    return DECODE_CACHE[key]

def _decode(raw_boxes, strides, image_dims, reg_max, *, dtype=np.float32):
    # boxes = None
    # print("raw_boxes", [b.shape for b in raw_boxes])
    res_shape = (1, sum([b.shape[1] * b.shape[2] for b in raw_boxes]), 4)
    boxes = np.zeros(res_shape, dtype=dtype)
    offset = 0
    for box_distribute, stride in zip(raw_boxes, strides):
        # create grid
        h, w = int(image_dims[0] / stride), int(image_dims[1] / stride)
        hw = h * w
        # assert box_distribute.shape[1:3] == (h, w), \
        #     f"box_distribute shape {box_distribute.shape[1:3]} != ({h}, {w})"

        center = _get_decode_cache(w, h, stride, dtype=dtype)

        # box distribution to distance
        reg_range = np.arange(reg_max + 1, dtype=dtype).reshape(1, 1, 1, -1)

        _shape = (-1, hw, 4, reg_max + 1)
        box_distance = _softmax(box_distribute.reshape(*_shape)).astype(dtype) * reg_range
        box_distance = box_distance.sum(axis=-1) * stride

        # we want to subtract these coordinates from the center
        # so we need to flip the sign
        box_distance[:, :, :2] *= -1

        # print("box_distance1", box_distance.shape, box_distance.dtype)
        decode_box = center[None] + box_distance

        # print("decode_box", decode_box.shape, decode_box.dtype)
        xs0 = decode_box[:, :, 0]
        ys0 = decode_box[:, :, 1]
        xs1 = decode_box[:, :, 2]
        ys1 = decode_box[:, :, 3]

        boxes[:, offset:offset + hw, 0] = (xs0 + xs1) / 2
        boxes[:, offset:offset + hw, 1] = (ys0 + ys1) / 2
        boxes[:, offset:offset + hw, 2] = (xs1 - xs0)
        boxes[:, offset:offset + hw, 3] = (ys1 - ys0)
        offset += hw
        # xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        # boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes

def crop_mask(masks, boxes):
    """
    Zeroing out mask region outside of the predicted bbox.
    Args:
        masks: numpy array of masks with shape [n, h, w]
        boxes: numpy array of bbox coords with shape [n, 4]
    """

    _, h, w = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    xs1, ys1, xs2, ys2 = np.array_split(np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)

    _mask = np.zeros((h, w), dtype=masks.dtype)
    for k, (x1, y1, x2, y2) in enumerate(zip(xs1[:, 0], ys1[:, 0], xs2[:, 0], ys2[:, 0])):
        _mask[y1:y2, x1:x2] = 1
        masks[k] *= _mask
        _mask[:] = 0
    return masks


def process_mask(protos, masks_in, bboxes, shape):
    mh, mw, c = protos.shape
    masks = _sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape((-1, mh, mw))
    if not masks.shape[0]:
        return None

    ih, iw = shape
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW

    masks = cv2.resize(masks.transpose(1, 2, 0), # HWC
                       shape, interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[..., np.newaxis]
    return masks.transpose(2, 0, 1)  # HWC

def postproccess(outputs: T.List[np.array], *,
                 anchors: dict,
                 img_dims: tuple,
                 score_threshold: float = 0.5,
                 nms_iou_thresh: float = 0.5,
                 classes: int = 80
                 ):

    num_classes = classes
    image_dims = tuple(img_dims)
    strides = anchors["strides"][::-1]
    reg_max = anchors["regression_length"]
    score_thres = score_threshold
    iou_thres = nms_iou_thresh

    t0 = time.time()
    decoded_boxes = _decode(outputs[:7:3], strides, image_dims, reg_max)
    dec_t = time.time() - t0

    # unpack data
    t0 = time.time()
    proto_data = outputs[9]
    n_masks = proto_data.shape[-1]


    scores = _unpack(outputs[1:8:3], num_classes)
    coeffs = _unpack(outputs[2:9:3], n_masks)


    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1), dtype=np.float32)
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)
    # print("scores", scores_obj.dtype)

    # re-arrange predictions for nms
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    unpack_t = time.time() - t0
    # print("preds", predictions.dtype)

    t0 = time.time()
    nms_results = seg_post.non_max_suppression(predictions,
                                               conf_thres=score_thres,
                                               iou_thres=iou_thres,
                                               multi_label=True)
    nms_t = time.time() - t0
    outputs = []
    image_size = np.tile(image_dims, 2)
    proc_t = 0

    # for key, arr in nms_res[0].items():
    #     print(key, arr.shape, arr.dtype)

    for protos, nms_res in zip(proto_data, nms_results):
        t0 = time.time()
        masks = process_mask(protos, nms_res["mask"], nms_res["detection_boxes"], image_dims)
        proc_t += time.time() - t0

        outputs.append(dict(
            mask=masks if masks is not None else [],
            detection_scores=nms_res["detection_scores"],
            detection_classes=nms_res["detection_classes"].astype(np.int32),
            detection_boxes=(nms_res["detection_boxes"] / image_size).astype(np.float32),
        ))

    logger.debug(f"[hailo - postprocess times] unpack: {unpack_t:.3f}s | nms: {nms_t:.3f}s | decode: {dec_t:.3f}s | mask process: {proc_t:.3f}s")
    return outputs
