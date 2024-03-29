from typing import AnyStr, Dict, Callable, List
import torch
from torch import nn
from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_model import BaseModel
global_hooks: Dict[AnyStr, List[Callable]] = {}


class DefaultModel(nn.Module, BaseModel):
    """ Use and/or override torch functions if you need to """

    def __init__(self, configs: Dict):
        super().__init__()
        self.device = TrainerContainer.device
        self.__dict__.update(configs.get('special_inputs', {}))
        self.configs = configs
        self.add_hooks()

    def register_pre_hook(self, name: AnyStr, hook: Callable) -> None:
        self.register_hook(f'before_{name}', hook)

    def register_hook(self, name: AnyStr, hook: Callable) -> None:
        """ Addes hooks before or after a specific function """
        if name in global_hooks:
            global_hooks[name].append(hook)
        else:
            global_hooks[name] = [hook]

    def register_post_hook(self, name: AnyStr, hook: Callable) -> None:
        self.register_hook(f'after_{name}', hook)

    def add_hooks(self):
        """ use register_pre_hook and register_post_hook to add hooks around
            main train loop functions.

            See trainer functions decorated with "run_hooks" function in train loop
        """
        pass

    def set_layers(self) -> List[nn.Module]:
        """ Creates a structure of dense layers
            specified under special_inputs
        """
        index = 1
        layers_sizes = getattr(self, 'layers_sizes', [10, 10, 10])
        # assert len(layers_sizes) > 1, 'number of layers defined in config file ' \
        #                         'in special_inputs mut be at least 1+'
        if layers_sizes == 0 or len(layers_sizes) == 0:
            layer1 = nn.Linear(self.input_dim, self.n_outputs)
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
        layers = [l.to(self.device) for l in layers]
        self.apply(init_weights)
        return layers

    def set_batches(self):
        """ Creates a structure of batches layers
            specified under special_inputs
        """
        batch_norms = []
        if hasattr(self, 'batchnorm') and getattr(self, 'batchnorm'):
            print('adding batchnorm layers to the NN')
            index = 0
            layers_sizes = getattr(self, 'layers_sizes', [10, 10, 10])
            if len(layers_sizes) > 0:
                for layer_size in layers_sizes[:-1]:
                    batch_norm = nn.BatchNorm1d(layer_size)  # nn.Linear(layer_size, layers_sizes[index - 1])
                    setattr(self, f'batchnorm{index}', batch_norm)
                    batch_norms.append(batch_norm)
                    index += 1
            else:
                batch_norms = []
            batch_norms = [b.to(self.device) for b in batch_norms]
            self.apply(init_weights)
        return batch_norms

    def model_epoch_logs(self) -> Dict:
        """ Called after each epoch
            Return: dict of whatever needs to be logged to tensorboard
        """
        return {}


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def run_hooks(func: Callable) -> Callable:
    """ inputs -> Before hook # 1 -> output
        output ->  Before hook # 2 -> output
        ...
        output ->  Before hook # N -> output

        inputs + output -> After hook # 1 -> output
        inputs + output -> After hook # 2 -> output
        ...
        inputs + output -> After hook # N output -> output
     """
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
                outputs = hook(*args, outputs)
        return outputs
    return func_hook


