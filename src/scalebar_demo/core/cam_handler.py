__all__ = ["CV2CamHandler"]

import cv2
import numpy as np
import structlog
from omegaconf import DictConfig

logger = structlog.get_logger()

class CV2CamHandler(object):

    def __init__(self, cfg:DictConfig):
        super(CV2CamHandler, self).__init__()

        self.cap = cv2.VideoCapture(cfg.cam.id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        logger.info("Camera capture size [requested]", width=cfg.cam.width, height=cfg.cam.height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.cam.fps)

    def read(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()
