import numpy as np
import structlog
import cv2
import typing as T

from omegaconf import DictConfig
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from collections import deque

from .base import BaseWindow
from ...cam_handler import CV2CamHandler
from .. import widgets

logger = structlog.get_logger()

class MainWindow(BaseWindow):

    def __init__(self, cfg: DictConfig, cam: CV2CamHandler, proc_func: T.Callable):
        super().__init__(cfg)
        logger.info("Create MainWindow")

        logger.info("Camera Module", cls=type(cam).__name__)

        center = QtWidgets.QWidget(parent=self)
        layout = QtWidgets.QVBoxLayout()
        center.setLayout(layout)

        # center.setStyleSheet("background-color: rgba(0,0,0, 127); border: 2px solid yellow;")

        self.header = header    = widgets.Header(cfg, parent=center)
        self.camera = camera    = widgets.CameraWidget(cfg, cam, parent=center)
        self.proc = proc        = widgets.FrameProcessor(cfg, proc_func=proc_func, parent=center)
        # self.extra_window       = windows.HighlightWindow(parent=self)
        # model_selection = widgets.ModelSelection(models=self.models, parent=center)

        # logger.info("Header size: ", size=header.sizeHint())
        logger.info("Camera size: ", size=camera.sizeHint())
        logger.info("Center size: ", size=center.sizeHint())


        layout.addWidget(header, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(camera, 4)
        # layout.addWidget(model_selection, 1)

        self.prediction_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.prediction_label)

        self.load_logos(cfg)

        self._proc_fps = deque(maxlen=30)
        self._cam_fps = deque(maxlen=30)
        self._scales = deque(maxlen=30)

        camera.frame_ready.connect(proc)
        camera.fps_update.connect(self.update_cam_fps)
        # camera.result.clicked.connect(self.extra_window.show)

        proc.fps_update.connect(self.update_proc_fps)
        proc.result_ready.connect(self.result_ready)
        proc.prediction_ready.connect(self.show_prediction)

        with open(cfg.root / cfg.assets_path.css / cfg.css_styles.main) as style:
            self.setStyleSheet(style.read())

        self.post_init(center, title="ScaleBar Demo")

    def closeEvent(self, event):
        # self.extra_window.close()

        # if self.send_window is not None:
        #     self.send_window.close()

        self.camera.close()
        super(MainWindow, self).closeEvent(event)

    def load_logos(self, cfg: DictConfig):
        # prepare the logo overlay!
        w, h = cfg.cam.width, cfg.cam.height
        self.overlay = np.zeros((h, w, 4), dtype=np.uint8)
        self.ov_mask = np.zeros((h, w), dtype=np.uint8)

        # just load the logos for the overlay which are enabled
        logos: list[DictConfig] = [logo for logo in cfg.logos if logo.enabled]

        for logo in logos:
            logo_img = cv2.imread(str(cfg.root / cfg.assets_path.logos / logo.file), cv2.IMREAD_UNCHANGED)
            has_alpha = logo_img.shape[-1] == 4

            if not has_alpha:
                logger.warning("Loaded logo has no alpha channel", logo=logo.file)

            offset_h, offset_w = logo.offset
            logo_img = cv2.resize(logo_img, None, fx=logo.scale, fy=logo.scale, interpolation=cv2.INTER_LINEAR_EXACT)
            mark_h, mark_w, _ = logo_img.shape

            if logo.position == "bottom-right":
                padding_h = (h - mark_h - offset_h, offset_h)
                padding_w = (w - mark_w - offset_w, offset_w)
            elif logo.position == "bottom-left":
                padding_h = (h - mark_h - offset_h, offset_h)
                padding_w = (offset_w, w - mark_w - offset_w)
            elif logo.position == "top-right":
                padding_h = (offset_h, h - mark_h - offset_h)
                padding_w = (w - mark_w - offset_w, offset_w)
            elif logo.position == "top-left":
                padding_h = (offset_h, h - mark_h - offset_h)
                padding_w = (offset_w, w - mark_w - offset_w)
            else:
                logger.warning("Unknown position location for logo", pos=logo.position, file=logo.file)
                continue

            padding = (padding_h, padding_w, (0, 0))
            logo_img = np.pad(logo_img, padding, mode="constant")
            logo_img = cv2.GaussianBlur(logo_img, ksize=(3, 3), sigmaX=0.5)

            if has_alpha:
                mask = logo_img[..., -1] >= 200
                self.overlay[mask] = logo_img[mask]
            else:
                mask = logo_img > 0
                self.overlay[mask,:3] = logo_img[mask]
            self.ov_mask = np.logical_or(self.ov_mask, mask)
        self.overlay_to_add = np.sum(self.ov_mask) > 0

    def show_prediction(self, predictions):
        self.prediction_label.setText(predictions[0, 0])
        # self.extra_window.title.setText("Predicted Class: {}".format(predictions[0, 0]))

    def result_ready(self, img):
        if isinstance(img, tuple):
            img, scale = img
            if scale is not None:
                self._scales.append(scale)
                self.update_status_bar()

        img = self.postprocess(img)
        self.camera.result.set_image(img)
        # self.extra_window.result.set_image(img)

    def postprocess(self, img):
        return self.add_overlay(img)

    def add_overlay(self, img: np.ndarray) -> np.ndarray:
        if not self.overlay_to_add:
            return img
        img[self.ov_mask] = self.overlay[self.ov_mask, :3]
        return img


    def update_proc_fps(self, fps):
        self._proc_fps.append(fps)
        self.update_status_bar()

    def update_cam_fps(self, fps):
        self._cam_fps.append(fps)
        self.update_status_bar()

    def update_status_bar(self):
        msgs = []

        if self._cam_fps:
            mean_fps = sum(self._cam_fps) / len(self._cam_fps)
            msgs.append("Camera: {:>7.1f} FPS".format(mean_fps))

        if self._proc_fps:
            mean_fps = sum(self._proc_fps) / len(self._proc_fps)
            msgs.append("Processing: {:>7.1f} FPS".format(mean_fps))

        if self._scales:
            mean_scale = sum(self._scales) / len(self._scales)
            msgs.append("Estimated scale: {:>7.1f} px / cm".format(10 * mean_scale))
        # msgs.append(self.hotkey_text)

        self.statusBar().showMessage(" | ".join(msgs))
