import numpy as np
from typing import Callable
from modules.transformers.base_transformers.base_transformer import BaseTransformer


class DefaultTransformer(BaseTransformer):
    """ A preprocessing function on a feature """

    def __init__(self):
        self._transformer = self.set_transformer()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ runs a transformer """
        return self.transformer(vector)

    def set_transformer(self) -> Callable:
        """ sets _transformer to default function that just passes input """
        return lambda x: x

    @property
    def transformer(self) -> Callable:
        """ a function that processes a feature """
        return self._transformer


