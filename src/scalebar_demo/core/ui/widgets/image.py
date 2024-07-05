__all__ = ["ImageWidget"]

import cv2
import numpy as np
import structlog
import pyqtgraph as pg

from PyQt5 import QtCore, QtGui

logger = structlog.get_logger()

class ImageWidget(pg.GraphicsLayoutWidget):
    clicked = QtCore.pyqtSignal()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        super().mouseDoubleClickEvent(event)
        self.clicked.emit()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(0)

        self.vb: pg.ViewBox = self.addViewBox()
        self.vb.setAspectLocked(True)
        p = QtGui.QPalette()
        self.vb.setBackgroundColor(p.window().color())
        self.ii = pg.ImageItem()
        self.ii.setAutoDownsample(False)
        self.vb.addItem(self.ii)
        self.img = None

    def set_image(self, image: np.ndarray):
        self.img = image
        temp = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), 0)
        # temp = cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        self.ii.setImage(temp)

    def link(self, vb_: pg.ViewBox) -> None:
        self.vb.setXLink(vb_)
        self.vb.setYLink(vb_)

    def reset(self) -> None:
        self.vb.autoRange(padding=0.0)
