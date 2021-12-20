import numpy as np
import pandas as pd
import os
from modules.helpers.namer import Namer
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from typing import Dict, List
import torch
from utils.common import setup_imports, unpickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


@registry.register_wrapper('torch_wrapper')
class TorchWrapper(DefaultWrapper):
    """ Any neural net model wrapper in pytorch """

    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.output_function = self.get_output_function()

    def get_classifier(self, configs):
        if configs.get('trainer', {}).get('resume', False):
            name = Namer.model_name(configs.get('model'))
            folder = CLASSIFIERS_DIR
            model_path = f'{folder}/{name}.pkl'
            if os.path.isfile(model_path):
                model = unpickle_obj(model_path)
                print(f'resumed {name}')
            else:
                raise ValueError(f'cannot resume model {name}'
                                 f' - no checkpoint exist in folder {folder}')
        else:
            setup_imports()
            model = registry.get_model_class(
                configs.get('model').get('name')
            )(configs)
        return model

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        """ returns probabilities, is used in prediction step.
            Uses only certain features that were used during training """

        examples = self.filter_features(examples)
        return self.clf.predict(torch.FloatTensor(
                    examples
                )).detach().numpy()

    def predict(self, examples: torch.FloatTensor) -> torch.Tensor:
        """ returned to metrics or predict_proba in prediction step """
        return self.clf.predict(examples)

    def forward(self, examples: torch.FloatTensor):
        """ returns outputs, not probs, is used in train """
        return self.clf.forward(examples)

    def train(self) -> None:
        self.clf.train()

    def eval(self) -> None:
        self.clf.eval()

    def get_output_function(self) -> torch.nn.Module:
        """ member field, activation function defined in __init__ """
        f = self.configs.get('model').get('activation_function',
                                          {'name': 'LogSoftmax', 'dim': 1})
        return getattr(torch.nn, f.get('name'))(dim=f.get('dim'))

    def parameters(self):
        return self.clf.parameters()

    def before_iteration_train(self,  *args, **kwargs) -> None:
        """ runs before optimizer.zero_grad() """
        self.default_hooks(self.before_iteration_train.__name__, *args, **kwargs)

    def before_iteration_valid(self,  *args, **kwargs):
        self.default_hooks(self.before_iteration_valid.__name__, *args, **kwargs)

    def end_iteration_train(self, *args, **kwargs):
        self.default_hooks(self.end_iteration_train.__name__, *args, **kwargs)

    def end_iteration_valid(self, *args, **kwargs):
        self.default_hooks(self.end_iteration_valid.__name__, *args, **kwargs)

    def before_epoch_train(self, *args, **kwargs):
        self.default_hooks(self.before_epoch_train.__name__, *args, **kwargs)

    def before_epoch_valid(self, *args, **kwargs):
        self.default_hooks(self.before_epoch_valid.__name__, *args, **kwargs)

    def after_epoch_train(self, *args, **kwargs):
        self.default_hooks(self.after_epoch_train.__name__, *args, **kwargs)

    def after_epoch_valid(self, *args, **kwargs):
        self.default_hooks(self.after_epoch_valid.__name__, *args, **kwargs)

    def default_hooks(self, name, *args, **kwargs):
        hooks = self.clf.hooks[name]
        for hook in hooks:
            hook(self, *args, **kwargs)

