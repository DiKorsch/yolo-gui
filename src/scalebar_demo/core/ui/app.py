
from omegaconf import DictConfig
from PyQt5.QtWidgets import QApplication

from ..cam_handler import CV2CamHandler
from .windows import MainWindow


class MainApp(QApplication):
	def __init__(self, cfg: DictConfig, cam: CV2CamHandler, *args, **kwargs):
		super(MainApp, self).__init__(*args, **kwargs)
		self.window = MainWindow(cfg, cam)
		self.window.show()
