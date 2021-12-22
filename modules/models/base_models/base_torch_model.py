from typing import AnyStr, Dict, Callable, OrderedDict, Type
from torch import nn

from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_model import BaseModel
from utils.constants import CLASSIFIERS_DIR


global_hooks: Dict[AnyStr, Callable] = OrderedDict()


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs):
        super().__init__()
        self.device = TrainerContainer.device
        self.__dict__.update(configs.get('special_inputs'))
        self.configs = configs
        self.add_hooks()

    def register_pre_hook(self, func_name, hook):
        self.register_hook(f'before_{func_name}', hook)

    def register_hook(self, name, hook):
        if name in global_hooks:
            global_hooks[name].append(hook)
        else:
            global_hooks[name] = [hook]

    def register_post_hook(self, func_name, hook):
        self.register_hook(f'after_{func_name}', hook)

    def add_hooks(self):
        pass


def run_hooks(func):
    def func_hook(self, inputs):
        b_name = f'before_{func.__name__}'
        if b_name in global_hooks:
            pre_hooks = global_hooks[b_name]
            for hook in pre_hooks:
                inputs = hook(self.wrapper.clf, inputs)
        outputs = func(self, inputs)

        a_name = f'after_{func.__name__}'
        if a_name in global_hooks:
            post_hooks = global_hooks[a_name]
            for hook in post_hooks:
                outputs = hook(self.wrapper.clf, outputs)
        return outputs
    return func_hook


