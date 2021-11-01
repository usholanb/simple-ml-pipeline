from torch import nn
import torch.nn.functional as F
from modules.models.base_models.torch_model import BaseTorchModel
from utils.registry import registry


@registry.register_model('custom_net')
class CustomNetModel(BaseTorchModel):
    def __init__(self, input_dim):
        super(CustomNetModel, self).__init__()
        prev_size = input_dim
        layers = self.configs.get('model').get('dense_layers', 3)
        for i, size in enumerate(layers):
            setattr(self, f'layer{i+1}', nn.Linear(prev_size, size))
            prev_size = size
        setattr(self, f'layer{len(layers)}', nn.Linear(prev_size, len(self.labels_type)))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    # def parameters(self, *args, **kwargs):
    #     return self.parameters()
