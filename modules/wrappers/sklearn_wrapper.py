from typing import Dict, Any
import numpy as np
import pandas as pd
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import get_outside_library
from utils.registry import registry


@registry.register_wrapper('sklearn')
class SKLearnWrapper(DefaultWrapper):
    """ Any sklearn model """

    def get_classifier(self, configs: Dict):
        hps = configs.get('special_inputs', {})
        clf = get_outside_library(self.configs.get('model').get('name'))(**hps)
        poly = self.configs.get('model').get('poly', None)
        if poly is not None:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            clf = make_pipeline(PolynomialFeatures(poly), clf)
        return clf

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        """ filters in needed features and makes prediction  """
        examples = self.filter_features(examples)
        return self.predict(examples)

    def predict(self, examples: pd.DataFrame) -> np.ndarray:
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        examples = examples.values
        if self.configs.get('trainer').get('label_type') == 'classification':
            result = np.zeros((len(examples), self.clf.n_outputs_))
            result[np.arange(len(examples)), self.clf.predict(examples).astype(int)] = 1
        else:
            result = self.clf.predict(examples)
        return result

    def fit(self, inputs, targets) -> None:
        self.n_outputs = 1 if len(targets.shape) == 1 \
            else targets.shape[1]
        self.clf.fit(inputs, targets)

