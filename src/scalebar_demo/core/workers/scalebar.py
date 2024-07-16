import cv2
import numpy as np

from dataclasses import dataclass
from scalebar.core.image_wrapper import Images
from scalebar.core.estimation import Distances
from scalebar.core.bounding_box import BoundingBox
from scalebar.core.size import Size
from scalebar import utils as scalebar_utils


@dataclass
class Result:
    images: Images
    position: BoundingBox

    corners: np.ndarray = None
    px_per_square: float = None
    mm_per_square: float = None

    @property
    def scale(self) -> float:
        if self.px_per_square is None:
            return None
        return self.px_per_square / self.mm_per_square


    @classmethod
    def process(cls, frame: np.ndarray, scale_bar_size: Size, mm_per_square: float) -> 'Result':
        images = Images(frame, size=scale_bar_size)

        template_size = images.structure_sizes.template_size
        min_distance = images.structure_sizes.size

        match, template = scalebar_utils.match_scalebar(images.masked, template_size=template_size)
        position = scalebar_utils.detect_scalebar(match, enlarge=template_size)

        scalebar = position.crop(images.equalized)
        mask = position.crop(match)

        bin_crop = scalebar_utils.threshold(scalebar, mode=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners = cv2.goodFeaturesToTrack(bin_crop,
                                        maxCorners=50,
                                        qualityLevel=0.1,
                                        minDistance=5*min_distance,
                                        mask=mask)

        if corners is None:
            return cls(images=images, position=position)

        corners = corners[:, 0, ::-1].astype(int)
        distances = Distances(corners)
        px_per_square = distances.optimal_distance()

        if px_per_square is None:
            return cls(images=images, corners=corners, position=position,
                       mm_per_square=mm_per_square)


        return cls(images=images, corners=corners, position=position,
                   px_per_square=px_per_square, mm_per_square=mm_per_square)


class ScalebarProcessor:

    def __init__(self, size_per_square: float = 1.0, size: Size = Size.MEDIUM):
        self.mm_per_square = size_per_square
        self.scale_bar_size = size

    def estimate(self, frame: np.ndarray) -> Result:
        return Result.process(frame, self.scale_bar_size, self.mm_per_square)

    def __call__(self, frame: np.ndarray) -> np.ndarray:

        scbar = self.estimate(frame)
        res = frame.copy()

        if scbar.corners is None:
            return res, None

        ys, xs = scbar.corners.transpose(1, 0)
        for x, y in zip(xs, ys):
            res = cv2.circle(res, (x + scbar.position.x, y + scbar.position.y), 2, (0, 0, 255), -1)

        scale = scbar.scale
        if scale is not None:
            res = self.add_measure(res, scale, mm=10)

        return res, scale

    def add_measure(self, img, px_per_mm: float, mm: float) -> np.ndarray:
        height, width, *_ = img.shape
        middle_x = width // 2
        # Calculate the length in pixels based on px_per_mm and mm
        length_px = px_per_mm * mm

        # Draw a line in the middle of the image based on the calculated length
        middle_y = height // 2
        cv2.line(img, (middle_x - int(length_px/2), middle_y), (middle_x + int(length_px/2), middle_y), (255, 0, 0), 5)

        return img
