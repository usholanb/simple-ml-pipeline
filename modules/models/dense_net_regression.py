from typing import Dict
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.torch_model import BaseTorchModel
from utils.registry import registry


@registry.register_model('dense_net_regression')
class DenseNetModel(BaseTorchModel):
    def __init__(self, special_inputs):
        super(DenseNetModel, self).__init__()
        self.__dict__.update(special_inputs)
        self.set_layers()

    def forward(self, x) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        outputs = self.layer3(x).flatten()
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return x
