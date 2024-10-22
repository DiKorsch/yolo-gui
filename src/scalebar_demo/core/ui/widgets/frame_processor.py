__all__ = ["FrameProcessor"]
import multiprocessing.dummy as mp

import numpy as np
import typing as T
import time

from PyQt5 import QtCore, QtWidgets

from scalebar_demo import utils
from scalebar_demo.core.workers.base import BaseWorker


class FrameProcessor(QtWidgets.QWidget, utils.TicTocMixin):

    result_ready = QtCore.pyqtSignal(object)
    prediction_ready = QtCore.pyqtSignal(object)

    # def set_model(self, item, model_name):
    #     self._current_model = item

    # @property
    # def model(self):
    #     return self.models[self._current_model]

    # def keyPressEvent(self, event):
    #     for model in self.models.values():
    #         model.handle_key(event.key())

    def __init__(self, cfg, worker: BaseWorker, *args, **kwargs):
        super(FrameProcessor, self).__init__(*args, **kwargs)
        # self.models = models
        # # just the take the first model as the initial model
        # self._current_model = models[list(models.keys())[0]]

        self.pool = mp.Pool(1) if cfg.run_async else None

        self.result = None
        self.wait = cfg.wait_after_process
        assert callable(worker), "worker must be callable"
        self._worker = worker

    def process(self, *args, **kw):
        res = self._worker(*args, **kw)
        if self.wait is not None and self.wait > 0:
            time.sleep(self.wait)
        return res

    def __call__(self, frame: np.ndarray):
        # this called by CaptureThread.frame_ready.emit(frame)
        if self.pool is None:
            return self.result_ready.emit(self.process(frame))

        if self.result is None:
            self.tic()
            self.result = self.pool.apply_async(self.process, args=(frame,))

        if self.result is not None and self.result.ready():
            self.result_ready.emit(self.result.get())

            self.result = None
            self.toc()
