from abc import ABC, abstractmethod

import numpy as np
from dependency_injector.wiring import Provide
from typing import Dict
from modules.containers.train_container import TrainContainer


class BaseWrapper(ABC):

    @abstractmethod
    def get_classifier(self, hps: Dict):
        """ returns a model object created with external library """

    @abstractmethod
    def predict_proba(self, examples) -> np.ndarray:
        """ makes PROBABILITIES prediction on examples of dim N X M where N is number of
          examples and M number of features """

    @abstractmethod
    def predict(self, examples):
        """ returned to metrics or predict_proba in prediction step """
