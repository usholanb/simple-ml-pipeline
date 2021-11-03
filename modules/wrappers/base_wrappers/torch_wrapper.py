import numpy as np
from torch import nn
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from typing import Dict, List
import torch


class TorchWrapper(DefaultWrapper):
    """ Any neural net model wrapper in pytorch """

    def __init__(self, configs: Dict, label_types: List):
        super().__init__(configs, label_types)
        self.output_function = self.get_output_function()

    def predict_proba(self, examples) -> np.ndarray:
        """ returns probabilities, is used in prediction.
            Uses only certain features that were used during training """
        examples = self.filter_features(examples)
        return self.clf.predict(torch.FloatTensor(
                    examples
                )).detach().numpy()

    def forward(self, examples):
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
