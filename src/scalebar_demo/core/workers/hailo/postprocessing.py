
import numpy as np
import structlog
import cv2

from scalebar_demo.utils.cython import nms as c_nms

# Contents copied from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing
# the implementations have been slightly improved to increase FPS

logger = structlog.get_logger()

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _softmax(x, axis=-1):
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=-1, keepdims=True)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

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

def decode(raw_boxes, strides, image_dims, reg_max, *, dtype=np.float32):
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


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=32, multi_label=True):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Args:
        prediction: numpy.ndarray with shape (batch_size, num_proposals, 351)
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximal number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals
    Returns:
         A list of per image detections, where each is a dictionary with the following structure:
         {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
         }
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, 32)),
                    "detection_classes": np.zeros((0, 80)),
                    "detection_scores": np.zeros((0, 80)),
                }
            )
            continue

        # Confidence = Objectness X Class Score
        x[:, 5:] *= x[:, 4:5]

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label &= nc > 1
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        # per-class NMS
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

        keep = c_nms(preds, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        out = x[keep]
        scores = out[:, 4]
        classes = out[:, 5]
        boxes = out[:, :4]
        masks = out[:, 6:]

        out = {"detection_boxes": boxes, "mask": masks, "detection_classes": classes, "detection_scores": scores}

        output.append(out)

    return output


def mask_to_polygons(mask, threshold: float = 0.5):
    # mask = np.ascontiguousarray(mask)
    contours, hierarchy = cv2.findContours((mask >= threshold).astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = [x.flatten() + 0.5 for x in contours if len(x) >= 6]
    return res, has_holes
