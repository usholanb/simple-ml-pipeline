from abc import ABC, abstractmethod
import numpy as np
from utils.common import Singleton


class BaseTransformer(ABC):
    """ A preprocessing function on a feature """
    __metaclass__ = Singleton

    @abstractmethod
    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ applys a transformer on 1D or 2D array vector """



