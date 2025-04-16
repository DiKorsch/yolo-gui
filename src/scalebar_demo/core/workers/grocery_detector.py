import structlog
import numpy as np
import typing as T
import multiprocessing as mp
import time
import torch as th

from PyQt5 import QtWidgets
from functools import partial
from ultralytics.engine import results

from scalebar_demo.core.workers.detection import DetectionWorker

logger = structlog.get_logger()

class Grabber(mp.Process):

    def __init__(self, lock: mp.Lock, *args, **kwargs):
        self.lock = lock
        self.a, self.b = mp.Pipe()
        super().__init__(*args, **kwargs)
        self.reset()

    def grab(self, preds: results.Results):
        with self.lock:
            logger.debug("Grabbing")
            time.sleep(5)
            #### TODO: Implement the grabbing
            logger.debug("Grabbing done")
            self.reset()

    def reset(self):
        #### TODO: put me in initial position
        pass

    def run(self):
        while True:
            if self.b.poll():
                logger.debug("Receiving job...")
                preds = self.b.recv()
                logger.debug("Job received, sending to the grabber...")
                self.grab(preds)
            else:
                logger.debug("No job yet...")
            time.sleep(1)

    def set_job(self, preds: results.Results):
        logger.debug("Sending job...")
        self.a.send(preds)
        logger.debug("Job sent")



class GroceryDetector(DetectionWorker):

    def __init__(self, *args, **kwargs):
        self.reset()
        super().__init__(*args, **kwargs)
        self.grabbing = mp.Lock()
        self.grabber = Grabber(self.grabbing)
        self.grabber.start()

    def __del__(self):
        if hasattr(self, "grabber"):
            self.grabber.kill()
            self.grabber.join()

    def reset(self) -> None:
        self.selected = None
        self.preds = None

    def setup_controls(self, parent: QtWidgets.QWidget) -> None:
        assert parent.layout() is not None, "Parent widget must have a layout"

        self.center = QtWidgets.QWidget(parent=parent)
        self.layout = QtWidgets.QHBoxLayout()
        self.center.setLayout(self.layout)
        parent.layout().addWidget(self.center, 1)

        self.selection_label = QtWidgets.QLabel("Please select a class", parent=self.center)
        self.layout.addWidget(self.selection_label)
        self.buttons = []
        for i, name in sorted(self.model.names.items(), key=lambda x: x[0]):
            btn = QtWidgets.QPushButton(name.capitalize(), parent=self.center)
            self.buttons.append(btn)
            self.layout.addWidget(btn)

            btn.clicked.connect(partial(self.select_class, idx=i, name=name))

        self.grab_btn = QtWidgets.QPushButton("Grab", parent=self.center)
        self.grab_btn.clicked.connect(self.grab)
        self.layout.addWidget(self.grab_btn)

    def select_class(self, idx: int, name: str) -> None:
        if self.selected is not None and self.selected[0] == idx:
            logger.info(f"Unselecting {self.selected[1]}")
            self.reset()
            self.selection_label.setText("Please select a class")
            return

        logger.info(f"Selected {name} ({idx})")
        self.selected = (idx, name)
        self.selection_label.setText(f"Selected class: {name}")

    def grab(self) -> None:
        if self.selected is None:
            logger.info("No class selected")
            return
        if self.preds is None:
            logger.info("No predictions yet")
            return
        logger.info("sending job to grabber")
        self.grabber.set_job(self.preds.cpu())

    def predict(self, image: np.ndarray) -> T.List[results.Results]:
        all_res: results.Results = super().predict(image)

        if self.selected is None:
            return all_res

        new_res = []
        for res in all_res:
            boxes, masks = res.boxes, res.masks
            if boxes is None:
                return res

            new_boxes, new_masks = [], []
            for i, box in enumerate(boxes):
                if box.cls == self.selected[0]:

                    new_boxes.append(box.data)
                    new_masks.append(masks[i].data)


            new_res.append(results.Results(
                orig_img=res.orig_img,
                path=res.path,
                names=res.names,
                speed=res.speed,

                boxes=th.vstack(new_boxes) if new_boxes else None,
                masks=th.vstack(new_masks) if new_masks else None,
            ))
        return new_res



    def __call__(self, image: np.ndarray) -> np.ndarray:
        res = image.copy()

        acq = self.grabbing.acquire(block=False)
        if acq:
            try:
                self.preds: results.Results = self.predict(image)[0]
            finally:
                self.grabbing.release()

        if self.preds is None:
            return res

        return self.preds.plot(img=res)
