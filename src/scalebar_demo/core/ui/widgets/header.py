import structlog

from PyQt5 import QtCore, QtWidgets, QtGui

logger = structlog.get_logger()

class Header(QtWidgets.QWidget):

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setup_ui(cfg)
        self.setStyleSheet("background-color: rgba(255,255,255, 127);"
                           "border: 2px solid rgba(127, 127, 127, 255);"
                           "border-radius: 10px;")


    def setup_ui(self, cfg):

        layout = QtWidgets.QHBoxLayout()
        # layout.setContentsMargins(40, 40, 40, 40)
        logo_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("src/assets/logos/IAB-Logo_sm.png")
        logo_label.setPixmap(pixmap)
        logo_label.setContentsMargins(10, 10, 10, 10)
        # logo_label.setMinimumSize(pixmap.size())

        layout.addWidget(logo_label, 1)

        if cfg.get("info_text"):
            info_text = QtWidgets.QTextBrowser()
            with open(cfg.info_text, "r") as f:
                info_text.setText(f.read())
            info_text.setContentsMargins(40, 40, 40, 40)
            layout.addWidget(info_text, 2)
        else:
            layout.addStretch(2)

        self.setLayout(layout)
