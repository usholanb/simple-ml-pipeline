
from torch import nn
from abc import ABC, abstractmethod
from modules.models.base_models.base_model import BaseModel


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """





