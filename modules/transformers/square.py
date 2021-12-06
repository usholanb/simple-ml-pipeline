import numpy as np
from modules.transformers.base_transformers.default_transformer import DefaultTransformer
from utils.registry import registry


@registry.register_transformer('square')
class SquareTransformer(DefaultTransformer):

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ applys a transformer on 1D or 2D array vector """
        return vector ** 2
