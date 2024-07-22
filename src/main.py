import hydra
import sys
import structlog
import pyqtgraph as pg
import matplotlib as mpl
mpl.use('Agg')

from omegaconf import DictConfig
from pathlib import Path
from PyQt5 import QtCore

from scalebar_demo import core
from scalebar_demo.core import workers

logger = structlog.get_logger()

@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    cfg.root = Path(sys.argv[0]).parent

    pg.setConfigOption(opt="imageAxisOrder", value="row-major")
    cam = core.CV2CamHandler(cfg)

    proc = workers.PoseEstimator(threed=False)

    # proc = workers.SizeEstimator(
    #     size_per_square=cfg.scale.square_size,
    #     size=Size.get(cfg.scale.size),
    #     detector=cfg.detector,
    # )

    # proc = workers.ScalebarProcessor(size_per_square=cfg.scale.square_size, size=Size.get(cfg.scale.size))
    # proc = workers.DetectionWorker(cfg.detector)
    app = core.ui.MainApp(cfg, cam, proc, [])

    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        logger.info("Setting high DPI pixmaps")
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    sys.exit(app.exec_())


main()
