from typing import Dict

from torch import nn
from abc import ABC, abstractmethod


class BaseModel(object):

    @abstractmethod
    def forward(self, data):
        """
        passes inputs through the model
        returns: anything that is feed to right to loss
        """

    @abstractmethod
    def predict(self, data):
        """
        Used during prediction step
        """





