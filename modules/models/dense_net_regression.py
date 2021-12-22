from typing import Dict
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.base_torch_model import BaseTorchModel
from utils.registry import registry


@registry.register_model('dense_net_regression')
class DenseNetRegressionsModel(BaseTorchModel):
    def __init__(self, configs):
        super(DenseNetRegressionsModel, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs', {}))
        self.layers = self.set_layers()

    def forward(self, x) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        outputs = self.layers[-1](x).flatten()
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return x



