from typing import Dict
import numpy as np
import pandas as pd
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import get_outside_library
from utils.registry import registry


@registry.register_wrapper('sklearn')
class SKLearnWrapper(DefaultWrapper):
    """ Any neural net model in pytorch """

    def get_classifier(self, hps: Dict):
        return get_outside_library(self.configs.get('model').get('name'))(**hps)

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        """ filters in needed features and makes prediction  """
        examples = self.filter_features(examples)
        return self.clf.predict(examples)

    def predict(self, examples: np.ndarray) -> np.ndarray:
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        if self.configs.get('trainer').get('classification'):
            result = np.zeros((len(examples), len(self.label_types)))
            result[np.arange(len(examples)), self.clf.predict(examples).astype(int)] = 1
        else:
            result = self.clf.predict(examples)
        return result

    def fit(self, inputs, targets) -> None:
        self.clf.fit(inputs, targets)

