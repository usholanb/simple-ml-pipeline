from typing import Dict, AnyStr

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

    @abstractmethod
    def model_path(self) -> AnyStr:
        """ absolute path to the model """

    @abstractmethod
    def add_hooks(self):
        """ add hooks before and after main trainer functions

            For reference: look at your trainer functions
                that are decorated with hooks
         """





