from torch import nn
from modules.models.base_models.default_model import DefaultModel
from typing import Dict, List


class TorchModel(DefaultModel):
    """ Any neural net model in pytorch """

    # def __init__(self, configs: Dict, label_types: List):
    #     super(DefaultModel, self).__init__(configs, label_types)

    def predict(self, examples):
        return self.clf.forward(examples)

    def forward(self, examples):
        return self.clf.forward(examples)
