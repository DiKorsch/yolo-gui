import hydra
import sys
import pyqtgraph as pg

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from PyQt5 import QtCore

from scalebar_demo import core

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg.root = Path(sys.argv[0]).parent

    pg.setConfigOption(opt="imageAxisOrder", value="row-major")
    cam = core.CV2CamHandler(cfg)
    app = core.ui.MainApp(cfg, cam, [])

    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    sys.exit(app.exec_())


main()
