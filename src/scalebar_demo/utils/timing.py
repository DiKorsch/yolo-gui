__all__ = ["TicTocMixin"]

import time

from PyQt5.QtCore import pyqtSignal

class TicTocMixin(object):

	fps_update = pyqtSignal(float)

	def __init__(self, *args, **kwargs):
		super(TicTocMixin, self).__init__(*args, **kwargs)
		self.tic()

	def tic(self):
		self.t0 = time.time()

	def toc(self):
		t = time.time() - self.t0
		self.t0 = time.time()
		self.fps_update.emit(1 / t)
		return t
