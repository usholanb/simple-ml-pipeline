from typing import AnyStr

import torch
from torch import nn
from abc import ABC, abstractmethod
from modules.models.base_models.base_model import BaseModel
from utils.constants import CLASSIFIERS_DIR


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__dict__.update(configs.get('special_inputs'))
        self.configs = configs

    @property
    def model_name(self) -> AnyStr:
        m = self.configs.get('model')
        return f'{m.get("name")}_{m.get("tag")}'

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.model_name}.pkl'







