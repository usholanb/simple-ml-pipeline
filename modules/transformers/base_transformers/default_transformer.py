import numpy as np
from typing import Callable
from modules.transformers.base_transformers.base_transformer import BaseTransformer


class DefaultTransformer(BaseTransformer):
    """ A preprocessing function on a feature """

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ applys a transformer on 1D or 2D array vector """
        return vector



