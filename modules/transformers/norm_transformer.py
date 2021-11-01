import numpy as np
from modules.transformers.base_transformers.default_transformer import DefaultTransformer
from utils.registry import registry


@registry.register_transformer('norm')
class Norm(DefaultTransformer):
    def set_transformer(self):
        return lambda x: x / np.linalg.norm(x)

