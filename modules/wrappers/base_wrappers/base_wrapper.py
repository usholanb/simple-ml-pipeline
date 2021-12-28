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

    @abstractmethod
    def get_prediction_probs(self, data):
        """ makes PROBABILITIES prediction on examples of dim N X M where N is number of
          examples and M number of features """

    @abstractmethod
    def get_train_probs(self, data):
        """ returned to metrics or predict_proba in prediction step """
