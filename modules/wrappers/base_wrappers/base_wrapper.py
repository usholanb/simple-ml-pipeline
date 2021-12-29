from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, AnyStr


class BaseWrapper(ABC):

    @abstractmethod
    def model_path(self) -> AnyStr:
        """ Returns abs path to the wrapper """

    @abstractmethod
    def name(self) -> AnyStr:
        """ Returns full name of the wrapper """

    @abstractmethod
    def get_classifier(self, hps: Dict):
        """ returns a model object created with external library """

