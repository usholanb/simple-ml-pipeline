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
        layers_sizes = getattr(self, 'layers_sizes', [10, 10, 10])
        # assert len(layers_sizes) > 1, 'number of layers defined in config file ' \
        #                         'in special_inputs mut be at least 1+'
        if layers_sizes == 0 or len(layers_sizes) == 0:
            layer1 = nn.Linear(self.input_dim, len(self.label_types))
            layers = [layer1]
            setattr(self, f'layer{index}', layer1)
        elif len(layers_sizes) > 0:
            layer1 = nn.Linear(self.input_dim, layers_sizes[0])
            layers = [layer1]
            setattr(self, f'layer{index}', layer1)
            index += 1
            for layer_size in layers[1:-1]:
                layer = nn.Linear(layer_size[index], layers_sizes[index])
                setattr(self, f'layer{index}', layer)
                layers.append(layer)
                index += 1
            last_layer = nn.Linear(layers_sizes[-1], len(self.label_types))
            setattr(self, f'layer{index}', last_layer)
            layers.append(last_layer)
        else:
            layers = []
        return layers







