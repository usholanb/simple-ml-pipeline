from torch import nn
from modules.models.base_model import BaseModel
from modules.models.default_model import DefaultModel


class TorchModel(nn.Module, DefaultModel):
    """ Any neural net model in pytorch """
