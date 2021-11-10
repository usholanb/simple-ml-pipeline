from typing import Dict
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.base_torch_model import BaseTorchModel
from utils.registry import registry


@registry.register_model('dense_net')
class DenseNetModel(BaseTorchModel):
    def __init__(self, special_inputs):
        super(DenseNetModel, self).__init__()
        self.__dict__.update(special_inputs)
        self.layer1 = nn.Linear(self.input_dim, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, len(self.label_types))
        self.output_function = nn.LogSoftmax(dim=1)
        self.prediction_function = nn.Softmax(dim=1)

    def forward(self, x) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        outputs = self.output_function(self.layer3(x))
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return self.prediction_function(x)
