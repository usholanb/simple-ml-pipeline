from torch import nn
import torch.nn.functional as F
from modules.models.base_models.base_special_model import BaseSpecialModel
from utils.registry import registry


@registry.register_special_model('dense_net')
class DenseNetModel(BaseSpecialModel):
    def __init__(self, input_dim):
        super(DenseNetModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
