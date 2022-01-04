from typing import Dict, Any
import numpy as np
import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import get_outside_library, unpickle_obj
from utils.constants import CLASSIFIERS_DIR
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

    def get_prediction_probs(self, examples: pd.DataFrame) -> np.ndarray:
        """ Returns probs
            filters in needed features and makes prediction
        """
        examples = self.filter_features(examples)
        return self.get_train_probs(examples)

    def get_train_probs(self, examples: np.ndarray) -> np.ndarray:
        """ Returns probs
            makes prediction on pandas examples of dim N X M
            where N is number of examples and M number of features
        """
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

    def predict_dataset(self, configs, split_names) -> Dict:
        splits = {}
        data = CSVSaver().load(configs)
        const = configs.get('static_columns')
        for split_name in split_names:
            split = data[data['split'] == split_name]
            split_y = split.iloc[:, const.get('FINAL_LABEL_INDEX')].values
            split_x = split.iloc[:, len(const):]
            preds = self.get_prediction_probs(split_x)

            splits[split_name] = {f'{split_name}_preds': preds,
                                  f'{split_name}_ys': split_y}
        return splits
