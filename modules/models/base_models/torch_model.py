from torch import nn
from modules.models.base_models.default_model import DefaultModel


class TorchModel(nn.Module, DefaultModel):
    """ Any neural net model in pytorch """

    def predict(self, examples):
        return self.clf.forward(examples)

    def predict_proba(self, examples):
        return self.clf.forward(examples)
