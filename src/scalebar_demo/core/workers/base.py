import abc
import numpy as np


class BaseWorker(abc.ABC):
    """Base class for workers that process frames."""

    @abc.abstractmethod
    def load_model(self, *args, **kwargs):
        """Load the model."""
        self.model = None

    @abc.abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return the result."""
        pass
