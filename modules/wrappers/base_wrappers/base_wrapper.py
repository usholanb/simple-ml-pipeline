from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, AnyStr

import pandas as pd


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
        """ Returns probs
            filters in needed features and makes prediction
        """

    @abstractmethod
    def get_train_probs(self, data):
        """ Returns probs
        """
