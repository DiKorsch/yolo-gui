__all__ = ["CameraWidget"]

import structlog
from PyQt5 import QtCore, QtWidgets

from scalebar_demo import utils
from scalebar_demo.core.ui import widgets

logger = structlog.get_logger()

class CaptureThread(QtCore.QThread, utils.TicTocMixin):

    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, cam, flip_image, temp_smoothing: bool = True, *args, **kwargs):
        super(CaptureThread, self).__init__(*args, **kwargs)
        self.cam = cam
        self.flip_image = flip_image
        self.temp_smoothing = temp_smoothing
        self.smoothing_factor = 0.1

        self.avg_frame = None
        self._running = True

        parent = self.parent()
        assert parent is not None, "Set parent parameter!"

        self.fps_update.connect(parent.fps_update.emit)
        self.frame_ready.connect(parent.frame_ready.emit)

    def run(self):
        while self._running:
            self.tic()
            ret, frame = self.cam.read()
            if not ret:
                continue

            if self.flip_image:
                frame = frame[:, ::-1, :]#.copy()

            if self.temp_smoothing:
                if self.avg_frame is None:
                    self.avg_frame = frame
                else:
                    a, b = self.smoothing_factor, 1-self.smoothing_factor
                    frame = self.avg_frame = (a * self.avg_frame + b * frame).astype(frame.dtype)

            self.frame_ready.emit(frame)

            self.toc()


class CameraWidget(QtWidgets.QWidget, utils.TicTocMixin):
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, opts, cam, *args, **kwargs):
        super(CameraWidget, self).__init__(*args, **kwargs)

        self.frame  = widgets.ImageWidget(parent=self)
        self.result = widgets.ImageWidget(parent=self)
        self.result.link(self.frame.vb)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.frame)
        layout.addWidget(self.result)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

        self._thread = CaptureThread(cam, opts.flip_image, parent=self)
        self._thread.start()
        self.frame_ready.connect(self.frame.set_image)

    def reset(self) -> None:
        self.frame.reset()
        self.result.reset()

    def closeEvent(self, event):
        self._thread._running = False

        self._thread.quit()
        self._thread.wait()

        event.accept()
