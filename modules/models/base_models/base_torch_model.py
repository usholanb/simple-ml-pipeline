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
        self.hook_types = ['before_iteration_train', 'end_iteration_train',
                           'before_iteration_valid', 'end_iteration_valid',
                           'before_epoch_train', 'after_epoch_train',
                           'before_epoch_valid', 'after_epoch_valid']
        self.device = TrainerContainer.device
        self.__dict__.update(configs.get('special_inputs'))
        self.configs = configs
        self._hooks = {}
        for hook_name in self.hook_types:
            self.hooks[hook_name] = []
        self.add_hooks()

    def add_hooks(self):
        pass

    @property
    def hooks(self):
        return self._hooks

    @property
    def model_name(self) -> AnyStr:
        m = self.configs.get('model')
        return f'{m.get("name")}_{m.get("tag")}'

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.model_name}.pkl'




