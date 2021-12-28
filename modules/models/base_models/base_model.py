from typing import Dict, AnyStr, Tuple

from torch import nn
from abc import ABC, abstractmethod


class BaseModel(object):


    @abstractmethod
    def predict(self, data):
        """
        Used during prediction step
        """

    @abstractmethod
    def model_path(self) -> AnyStr:
        """ absolute path to the model """





