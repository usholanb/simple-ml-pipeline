import torch
from torch import nn
from abc import ABC, abstractmethod
from modules.models.base_models.base_model import BaseModel


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self):
        super().__init__()
        if torch.has_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def set_layers(self):
        index = 1
        layers = getattr(self, 'layers', [10, 10, 10])
        assert len(layers) > 1, 'number of layers defined in config file ' \
                                'in special_inputs mut be at least 1+'
        setattr(self, f'layer{index}', nn.Linear(self.input_dim, layers[0]))
        index += 1
        for layer_size in layers[1:-1]:
            setattr(self, f'layer{index}', nn.Linear(layer_size, layers[index]))
            index += 1
        setattr(self, f'layer{index}', nn.Linear(layers[-1], len(self.label_types)))






