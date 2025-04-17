
import cv2
import numpy as np
import pyximport
from scalebar_demo.utils.timing import TicTocMixin

def putText(image, text, coords, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), thickness=2):
    x0, y0, x1, y1 = coords
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cx, cy = (x0 + x1 - w) // 2, (y0 + y1 - (h + baseline)) // 2
    cv2.putText(image, text, (cx, cy), font, font_scale, color, thickness)
    return image

__all__ = [
    "TicTocMixin",
    "putText",
]

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
