from typing import AnyStr
import torch
from torch import nn

from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_model import BaseModel
from utils.constants import CLASSIFIERS_DIR


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs):
        super().__init__()
        self.device = TrainerContainer.device
        self.__dict__.update(configs.get('special_inputs'))
        self.configs = configs

    @property
    def model_name(self) -> AnyStr:
        m = self.configs.get('model')
        return f'{m.get("name")}_{m.get("tag")}'

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.model_name}.pkl'

    def before_epoch(self):
        pass

    def before_iteration_train(self, all_data):
        pass

    def before_iteration_valid(self, all_data):
        pass

    def after_iteration(self, all_data):
        pass

    def after_epoch(self, all_data):
        pass





