from typing import Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.default_model import DefaultModel
from utils.registry import registry


@registry.register_model('dense_net_regression')
class DenseNetRegressions(DefaultModel):
    def __init__(self, configs: Dict):
        super(DenseNetRegressions, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs', {}))
        self.layers = self.set_layers()
        self.batchnorms = self.set_batches()
        if hasattr(self, 'dropout'):
            self.drop_layer = nn.Dropout(self.dropout)

    def get_x_y(self, batch) -> Tuple:
        x, y = batch
        if self.n_outputs == 1:
            y = y.reshape(-1, self.n_outputs)
        return x, y

    def forward(self, data: Dict) -> torch.Tensor:
        x = data['x']
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.batchnorms:
                x = self.batchnorms[i](x)
            x = F.relu(x)
            if hasattr(self, 'drop_layer'):
                x = self.drop_layer(x)
        return self.layers[-1](x)

    def predict(self, x: Dict):
        x = self.forward(x)
        return x



