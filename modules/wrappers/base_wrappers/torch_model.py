from torch import nn
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from typing import Dict, List
import torch


class TorchWrapper(DefaultWrapper):
    """ Any neural net model in pytorch """

    def predict(self, examples):
        return self.clf.forward(examples)

    def forward(self, examples):

        return self.clf.forward(examples)

    def train(self):
        self.clf.train()

    def eval(self):
        self.clf.eval()

