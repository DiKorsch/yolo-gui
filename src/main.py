import hydra
import sys
import structlog
import logging
import pyqtgraph as pg
import matplotlib as mpl
mpl.use('Agg')

from omegaconf import DictConfig
from pathlib import Path
from PyQt5 import QtCore

from scalebar_demo import core
from scalebar.core.size import Size
from scalebar_demo.core import workers


structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = structlog.get_logger()

def new_frame_processor(cfg: DictConfig):
    if cfg.model.name == "PoseEstimator":
        return workers.PoseEstimator(threed=cfg.model.threed)
    elif cfg.model.name == "ScaleEstimator":
        scale_info = cfg.model.scale
        # proc = workers.ScalebarProcessor(size_per_square=scale_info.square_size,
        #                                  size=Size.get(scale_info.size))
        # proc = workers.DetectionWorker(cfg.model.weights)
        return workers.SizeEstimator(
            size_per_square=scale_info.square_size,
            size=Size.get(scale_info.size),
            detector=cfg.model.weights,
        )
    elif cfg.model.name == "Detector":
        return workers.DetectionWorker(
            snapshot=cfg.model.weights,
            extra_snapshots=cfg.model.extra_snapshots
        )

    elif cfg.model.name == "GroceryDetector":
        return workers.GroceryDetector(
            snapshot=cfg.model.weights,
            extra_snapshots=cfg.model.extra_snapshots
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    cfg.root = Path(sys.argv[0]).parent

    pg.setConfigOption(opt="imageAxisOrder", value="row-major")
    cam = core.init_cam_handler(cfg)

    proc = new_frame_processor(cfg)
    app = core.ui.MainApp(cfg, cam, proc, [])

    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        logger.info("Setting high DPI pixmaps")
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, on=True)

    sys.exit(app.exec_())


main()
