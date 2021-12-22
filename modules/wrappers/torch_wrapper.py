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
            if os.path.isfile(self.model_path):
                model = unpickle_obj(self.model_path)
                print(f'resumed {self.name}')
            else:
                raise ValueError(f'cannot resume model {self.name}'
                                 f' - no checkpoint exist in'
                                 f' folder {CLASSIFIERS_DIR}')
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

    def prepare_data(self, data):
        return self.clf.prepare_data(data)


