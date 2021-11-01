from abc import ABC, abstractmethod
import numpy as np


class BaseTransformer(ABC):
    """ A preprocessing function on a feature """

    @abstractmethod
    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ runs a transformer """

    @abstractmethod
    def set_transformer(self) -> None:
        """ sets _transformer """

    @property
    @abstractmethod
    def transformer(self) -> None:
        """ a function that processes a feature """


