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

    def model_path(self) -> AnyStr:
        m = self.configs.get('model')
        return f'{CLASSIFIERS_DIR}/{m.get("name")}_{m.get("tag")}.pkl'

    def before_epoch(self):
        pass

    def before_iteration(self, all_data):
        pass

    def after_iteration(self, all_data):
        pass

    def after_epoch(self, all_data):
        pass





