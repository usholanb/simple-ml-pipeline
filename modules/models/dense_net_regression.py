from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from modules.models.base_models.default_model import DefaultModel
from utils.registry import registry


@registry.register_model('dense_net_regression')
class DenseNetRegressions(DefaultModel):
    def __init__(self, configs: Dict):
        super(DenseNetRegressions, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs', {}))
        self.layers = self.set_layers()

    def get_x_y(self, batch) -> Tuple:
        return batch

    def forward(self, data: Dict) -> torch.Tensor:
        x = data['x']
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def predict(self, x: Dict):
        x = self.forward(x)
        return x



