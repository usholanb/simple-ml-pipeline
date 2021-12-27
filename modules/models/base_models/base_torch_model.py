from typing import AnyStr, Dict, Callable, OrderedDict, Type, List
from torch import nn

from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_model import BaseModel
from utils.constants import CLASSIFIERS_DIR


global_hooks: Dict[AnyStr, List[Callable]] = OrderedDict()


class BaseTorchModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs):
        super().__init__()
        self.device = TrainerContainer.device
        self.__dict__.update(configs.get('special_inputs', {}))
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
            for layer_size in layers_sizes[:-1]:
                layer = nn.Linear(layer_size, layers_sizes[index - 1])
                setattr(self, f'layer{index}', layer)
                layers.append(layer)
                index += 1
            last_layer = nn.Linear(layers_sizes[-1], self.n_outputs)
            setattr(self, f'layer{index}', last_layer)
            layers.append(last_layer)
        else:
            layers = []
        return layers


def run_hooks(func):
    def func_hook(self, *args):
        pre_inputs = args
        b_name = f'before_{func.__name__}'
        if b_name in global_hooks:
            pre_hooks = global_hooks[b_name]
            for hook in pre_hooks:
                pre_inputs = hook(*pre_inputs)

        outputs = func(self, *pre_inputs)

        a_name = f'after_{func.__name__}'
        if a_name in global_hooks:
            post_hooks = global_hooks[a_name]
            for hook in post_hooks:
                outputs = hook(*args, *outputs)
        return outputs
    return func_hook


