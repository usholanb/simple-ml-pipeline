import numpy as np
from modules.transformers.base_transformers.default_transformer import DefaultTransformer
from utils.registry import registry


@registry.register_transformer('ohe')
class Ohe(DefaultTransformer):
    def apply(self, vector: np.ndarray) -> np.ndarray:
        val_to_index = {val: index for index, val in enumerate(sorted(np.unique(vector)))}
        output = np.zeros((len(vector), len(val_to_index)))
        idx = np.array([val_to_index[val] for val in vector.tolist()])
        output[np.arange(len(vector)), idx] = 1
        return output

