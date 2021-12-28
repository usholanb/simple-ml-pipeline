from typing import Dict
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.default_model import DefaultModel
from utils.registry import registry


@registry.register_model('dense_net')
class DenseNet(DefaultModel):
    def __init__(self, configs):
        super(DenseNet, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs'))
        self.layers = self.set_layers()
        self.prediction_function = nn.Softmax(dim=1)
        self.output_function = nn.LogSoftmax(dim=1)

    def forward(self, x) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        outputs = self.layers[-1](x)
        outputs = self.output_function(outputs)
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return self.prediction_function(x)
