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







