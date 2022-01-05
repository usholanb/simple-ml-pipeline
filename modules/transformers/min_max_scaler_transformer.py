import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules.transformers.base_transformers.base_transformer import BaseTransformer
from utils.registry import registry


@registry.register_transformer('min_max_scaler')
class MinMaxScalerTransformer(BaseTransformer):

    def __init__(self, configs):
        super(MinMaxScalerTransformer, self).__init__(configs)
        self.scaler = MinMaxScaler()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """ applys a transformer on 1D or 2D array vector """
        return self.scaler.fit_transform(vector.reshape(-1, 1)).flatten()
