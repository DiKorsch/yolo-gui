
from PyQt5 import QtGui, QtWidgets
from omegaconf import DictConfig

class BaseWindow(QtWidgets.QMainWindow):

    def __init__(self, cfg: DictConfig=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if cfg is None:
            if self.parent() is None or getattr(self.parent(), "cfg", None) is None:
                raise Exception("Require cfg:DictConfig either as parameter or in parent object")
            cfg = self.parent().cfg

        self.cfg = cfg
        self.icon = QtGui.QIcon(
            str(cfg.root / cfg.assets_path.icons / cfg.icon)
        )


    def post_init(self, central_widget, title=""):
        self.setWindowTitle(title)
        self.setCentralWidget(central_widget)
        self.setMinimumSize(800, 600)
        self.setWindowIcon(self.icon)

    def keyPressEvent(self, event):
        super(BaseWindow, self).keyPressEvent(event)

        if self.parent() is not None:
            self.parent().keyPressEvent(event)
