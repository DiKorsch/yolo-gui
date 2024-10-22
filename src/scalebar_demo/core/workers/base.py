import abc
import numpy as np

from PyQt5 import QtWidgets


class BaseWorker(abc.ABC):
    """Base class for workers that process frames."""

    @abc.abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return the result."""
        pass

    def setup_controls(self, parent: QtWidgets.QWidget) -> None:
        """Set up the controls."""
        pass
