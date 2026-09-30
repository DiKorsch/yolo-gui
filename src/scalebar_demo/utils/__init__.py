
import cv2
import numpy as np
from scalebar_demo.utils.timing import TicTocMixin

def putText(image, text, coords, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), thickness=2):
    x0, y0, x1, y1 = coords
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cx, cy = (x0 + x1 - w) // 2, (y0 + y1 - (h + baseline)) // 2
    cv2.putText(image, text, (cx, cy), font, font_scale, color, thickness)
    return image

def nms(dets: np.ndarray, thresh: float) -> np.ndarray:
    """
    Non-Maximum Suppression (NMS)

    Parameters
    ----------
    dets : np.ndarray
        Array der Form (N, 5):
        [x1, y1, x2, y2, score]
    thresh : float
        IoU-Schwellwert.

    Returns
    -------
    np.ndarray
        Indizes der beibehaltenen Bounding Boxes.
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Scores absteigend sortieren
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype=np.int32)

    for _i in range(ndets):
        i = order[_i]

        if suppressed[i]:
            continue

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, ndets):
            j = order[_j]

            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)

            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]

__all__ = [
    "TicTocMixin",
    "putText",
    "nms",
]

try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
except ImportError:
    pass

