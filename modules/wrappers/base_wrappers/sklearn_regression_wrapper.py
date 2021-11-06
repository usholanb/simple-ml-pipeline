import numpy as np
import pandas as pd

from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper


class SKLearnRegressionWrapper(DefaultWrapper):
    """ Any neural net model in pytorch """

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        """ filters in needed features and makes prediction  """
        examples = self.filter_features(examples)
        return self.clf.predict(examples)

    def predict(self, examples: np.ndarray) -> np.ndarray:
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        return self.clf.predict(examples)

    def fit(self, inputs, targets) -> None:
        self.clf.fit(inputs, targets)


