import numpy as np
import structlog
import re
import cv2

try:
    import hailo_platform as HAILO
except ImportError:
    HAILO_AVAILABLE = False
else:
    HAILO_AVAILABLE = True

from scalebar_demo.core.workers.base import BaseWorker
from scalebar_demo.core.workers.hailo.result import Result

logger = structlog.get_logger()

class HailoWorker(BaseWorker):
    target = None

    def __init__(self, hef_path: str, conf: dict):
        assert HAILO_AVAILABLE, "Hailo platform is not available. Please install hailo_platform."
        devices = HAILO.Device.scan()

        self.target = HAILO.VDevice(device_ids=devices)

        hef = HAILO.HEF(hef_path)
        inputs = hef.get_input_vstream_infos()
        assert len(inputs) == 1, "Only one input is supported"
        input_info = self.input_info = inputs[0]
        self.prefix = re.match(r"([a-zA-Z0-9_]+)/.*", input_info.name).group(1)

        logger.info(f"[hailo] Input name: {input_info.name} | shape: {input_info.shape}")
        logger.info(f"[hailo] Estimated model prefix: {self.prefix}")

        config_params = HAILO.ConfigureParams.create_from_hef(hef, interface=HAILO.HailoStreamInterface.PCIe)
        self.netgroup = self.target.configure(hef, config_params)[0]
        self.netgroup_params = self.netgroup.create_params()

        self.conf = conf

    def __call__(self, image: np.ndarray) -> np.ndarray:

        _params = dict(configured_network=self.netgroup, quantized=False, format_type=HAILO.FormatType.FLOAT32)
        input_vstreams_params = HAILO.InputVStreamParams.make_from_network_group(**_params)
        output_vstreams_params = HAILO.OutputVStreamParams.make_from_network_group(**_params)
        X = np.expand_dims(cv2.resize(image, (640, 640)), axis=0).astype(np.float32)

        with HAILO.InferVStreams(self.netgroup, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with self.netgroup.activate(self.netgroup_params):
                # t0 = time.time()
                raw_detections = infer_pipeline.infer({self.input_info.name: X})
                # infer_t = time.time() - t0
                # t0 = time.time()
                result = Result.postprocess(raw_detections, prefix=self.prefix, **self.conf)[0]
                # post_t = time.time() - t0

        # t0 = time.time()
        final_res = result.plot(image,
                                thresh=self.conf["score_threshold"],
                                alpha=None)
        # plot_t = time.time() - t0
        # logger.info(f"[hailo] Inference time: {infer_t:.3f}s | Postprocess time: {post_t:.3f}s | Plot time: {plot_t:.3f}s")
        return final_res

    def __del__(self):
        if self.target:
            self.target.release()
            self.target = None
