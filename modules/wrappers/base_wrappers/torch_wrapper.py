from torch import nn
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from typing import Dict, List
import torch


class TorchWrapper(DefaultWrapper):
    """ Any neural net model wrapper in pytorch """

    def __init__(self, configs: Dict, label_types: List):
        super().__init__(configs, label_types)
        self.output_function = self.get_output_function()

    def predict(self, examples):
        if len(examples.shape) == 1:
            examples = examples.reshape(1, -1)
        return torch.nn.Softmax(dim=1)\
            (
                self.clf.forward(torch.FloatTensor(
                    examples[self.config.get('features_list')].values.astype(float)
                ))
            )

    def forward(self, examples):
        return self.clf.forward(examples)

    def train(self):
        self.clf.train()

    def eval(self):
        self.clf.eval()

    def get_output_function(self):
        """ member field, activation function defined in __init__ """
        f = self.config.get('model').get('activation_function',
                                         {'name': 'LogSoftmax', 'dim': 1})
        return getattr(torch.nn, f.get('name'))(dim=f.get('dim'))
