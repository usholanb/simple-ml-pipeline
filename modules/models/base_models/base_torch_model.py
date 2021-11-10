import torch
from torch import nn
from abc import ABC, abstractmethod
from modules.models.base_models.base_model import BaseModel


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__dict__.update(configs.get('special_inputs'))
        self.configs = configs






