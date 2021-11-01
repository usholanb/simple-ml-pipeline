from torch import nn
import torch.nn.functional as F
from modules.models.base_models.torch_model import BaseTorchModel
from utils.registry import registry


@registry.register_model('dense_net')
class DenseNetModel(BaseTorchModel):
    def __init__(self, special_inputs):
        super(DenseNetModel, self).__init__()
        self.__dict__.update(special_inputs)
        self.layer1 = nn.Linear(self.input_dim, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    # def parameters(self, *args, **kwargs):
    #     return self.parameters()
