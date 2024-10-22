__all__ = ["CV2CamHandler"]

import cv2
import numpy as np
import structlog
from omegaconf import DictConfig

logger = structlog.get_logger()

def init_cam_handler(cfg: DictConfig):
	handler = dict(
		cv2=CV2CamHandler,
		realsense=RealSenseCamHandler,
	).get(cfg.cam.handler)
	if handler is None:
		logger.error("Unknown camera handler", handler=cfg.cam.handler)
		raise ValueError(f"Unknown camera handler: {cfg.cam.handler}")
	return handler(cfg)

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

class RealSenseCamHandler:
	def __init__(self, cfg: DictConfig):
		super(RealSenseCamHandler, self).__init__()

		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.config.enable_stream(rs.stream.color, cfg.cam.width, cfg.cam.height, rs.format.bgr8, cfg.cam.fps)

		self.pipeline.start(self.config)

	def read(self):
		frames = self.pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		color_image = np.asanyarray(color_frame.get_data())

		return True, color_image
