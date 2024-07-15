import hydra
import sys
import cv2
import pyqtgraph as pg
import numpy as np

from omegaconf import DictConfig
from pathlib import Path
from PyQt5 import QtCore

from scalebar_demo import core
from scalebar.core.image_wrapper import Images
from scalebar.core.estimation import Distances
from scalebar.core.size import Size
from scalebar import utils as scalebar_utils

class Processor:

    def __init__(self, size_per_square: float = 1.0, size: Size = Size.MEDIUM):
        self.mm_per_square = size_per_square
        self.scale_bar_size = size

    def __call__(self, frame: np.ndarray) -> np.ndarray:

        images = Images(frame, size=self.scale_bar_size)

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


        # masked = scalebar_utils.hide_non_roi(images.binary, self.scale_bar_size.value / 2, 127, location=None)
        # res = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        res = frame.copy()
        cropped = False

        if corners is None:
            return res, None

        corners = corners[:, 0, ::-1].astype(int)
        ys, xs = corners.transpose(1, 0)

        for x, y in zip(xs, ys):
            if cropped:
                res = cv2.circle(res, (x, y), 2, (0, 0, 255), -1)
            else:
                res = cv2.circle(res, (x + position.x, y + position.y), 2, (0, 0, 255), -1)

        distances = Distances(corners)
        px_per_square = distances.optimal_distance()

        px_per_mm = None
        if px_per_square is not None:
            px_per_mm = px_per_square / self.mm_per_square
            res = self.add_measure(res, px_per_mm, mm=10)
        return res, px_per_mm

    def add_measure(self, img, px_per_mm: float, mm: float) -> np.ndarray:
        height, width, *_ = img.shape
        middle_x = width // 2
        # Calculate the length in pixels based on px_per_mm and mm
        length_px = px_per_mm * mm

        # Draw a line in the middle of the image based on the calculated length
        middle_y = height // 2
        cv2.line(img, (middle_x - int(length_px/2), middle_y), (middle_x + int(length_px/2), middle_y), (255, 0, 0), 5)

        return img

@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    cfg.root = Path(sys.argv[0]).parent

    pg.setConfigOption(opt="imageAxisOrder", value="row-major")
    cam = core.CV2CamHandler(cfg)
    proc = Processor(size_per_square=cfg.scale.square_size, size=Size.get(cfg.scale.size))
    app = core.ui.MainApp(cfg, cam, proc, [])

    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    sys.exit(app.exec_())


main()
